#
#    This file is part of Fire Detection UAS.
#
#    Fire Detection UAS is free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Fire Detection UAS is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Fire Detection UAS.  If not, see
#    <https://www.gnu.org/licenses/>.
#

from enum import Enum

import numpy as np
import cv2


class Risk(Enum):
    NONE = 0
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5

    def __lt__(self, other):
        return self.value < other.value


# Colors in OpenCV are BGR
COLOR_TO_RISK = {
    (254, 241, 197, 255): Risk.NONE,
    (216, 216, 216, 255): Risk.NONE,
    (255, 0, 0, 255): Risk.VERY_LOW,
    (0, 255, 0, 255): Risk.LOW,
    (0, 255, 255, 255): Risk.MEDIUM,
    (0, 127, 255, 255): Risk.HIGH,
    (0, 0, 255, 255): Risk.VERY_HIGH,
}

RISK_TO_COLOR = {
    Risk.NONE: (255, 255, 255, 255),
    Risk.VERY_LOW: (255, 0, 0, 255),
    Risk.LOW: (0, 255, 0, 255),
    Risk.MEDIUM: (0, 255, 255, 255),
    Risk.HIGH: (0, 127, 255, 255),
    Risk.VERY_HIGH: (0, 0, 255, 255),
}


def read_image_with_alpha(file_):
    img = cv2.imread(file_, cv2.IMREAD_UNCHANGED)
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))


def blend(img1, img2):
    b1, g1, r1, alpha1 = cv2.split(img1)
    b2, g2, r2, alpha2 = cv2.split(img2)
    alpha = alpha2.astype(float) / 255
    b = b2 * alpha + b1 * (1 - alpha)
    g = g2 * alpha + g1 * (1 - alpha)
    r = r2 * alpha + r1 * (1 - alpha)
    alpha = np.maximum(alpha1, alpha2)
    return cv2.merge(
        (b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8), alpha))


def window_is_open(window_name):
    try:
        cv2.getWindowProperty(window_name, 0)
    except cv2.error:
        return False
    return True


def show_step(window_name, image, delay):
    cv2.imshow(window_name, image)
    return cv2.waitKey(delay)


def create_grid(height, width, cell_size):
    grid = np.zeros((height, width, 4), dtype=np.uint8)
    for i in range(0, height, cell_size):
        cv2.line(grid, (0, i), (width, i), (0, 0, 0, 255), 1)
    for i in range(0, width, cell_size):
        cv2.line(grid, (i, 0), (i, height), (0, 0, 0, 255), 1)
    return grid


def average_cells(img, cell_size):
    height, width, channels = img.shape
    rows = height // cell_size
    columns = width // cell_size
    cells = np.zeros((rows, columns, channels), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            irdi_sum = np.zeros(channels, dtype=int)
            for ii in range(cell_size):
                for jj in range(cell_size):
                    irdi_sum += img[i*cell_size + ii][j*cell_size + jj]
            cells[i][j] = irdi_sum // cell_size**2
    return cells


def create_rasterized_image(cells, cell_size):
    rows, columns, channels = cells.shape
    height = rows * cell_size
    width = columns * cell_size
    result = np.zeros((height, width, channels), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            for ii in range(cell_size):
                for jj in range(cell_size):
                    result[i*cell_size + ii][j*cell_size + jj] = cells[i][j]
    return result


def classify_cells(cells):
    rows, columns, channels = cells.shape
    classified_cells = np.zeros((rows, columns), dtype=Risk)
    for i in range(rows):
        for j in range(columns):
            limits = rows, columns
            classified_cells[i][j] = __get_risk(cells[i][j])
    return classified_cells


def __get_risk(cell):
    distances_and_risks = []
    for color, risk in COLOR_TO_RISK.items():
        distance = sum(pow(color[i] - cell[i], 2) for i in range(3))
        distances_and_risks.append((distance, risk))
    distances_and_risks.sort()
    return distances_and_risks[0][1]


def risk_cells_to_color_cells(cells):
    rows, columns = cells.shape
    risk_color_cells = np.zeros((rows, columns, 4), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            risk_color_cells[i][j] = RISK_TO_COLOR[Risk(cells[i][j])]
    return risk_color_cells


def smooth(cells, factor):
    """
    Smooth cells based on the median.
    """
    rows, columns = cells.shape
    smoothed_cells = np.zeros((rows, columns), dtype=cells.dtype)
    for i in range(rows):
        for j in range(columns):
            region_cells = []
            for ii in range(-factor, factor + 1):
                for jj in range(-factor, factor + 1):
                    row = (i + ii) % rows
                    column = (j + jj) % columns
                    region_cells.append(cells[row][column])
            region_cells.sort()
            median_index = len(region_cells) // 2
            smoothed_cells[i][j] = region_cells[median_index]
    return smoothed_cells
