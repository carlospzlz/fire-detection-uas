#!/usr/bin/env python

import argparse
from enum import Enum
from math import pow

import numpy as np
import cv2


FILE = 'maps/irdi_map_2019_10_08.jpg'
CELL_SIZE = 10


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


def parse_args():
    """
    Parses map and cell size.
    """
    parser = argparse.ArgumentParser(
        description='Experiment with Galicia\'s IRDI map.')
    parser.add_argument('map', type=str, help='map file')
    parser.add_argument('cell_size', type=int, help='cell size')
    return parser.parse_args()


def read_image_with_alpha(file_):
    img = cv2.imread(file_, cv2.IMREAD_UNCHANGED)
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))


def create_grid(height, width, cell_size):
    grid = np.zeros((height, width, 4), dtype=np.uint8)
    for i in range(0, height, cell_size):
        cv2.line(grid, (0, i), (width, i), (0, 0, 0, 255), 1)
    for i in range(0, width, cell_size):
        cv2.line(grid, (i, 0), (i, height), (0, 0, 0, 255), 1)
    return grid


def blend(img1, img2):
    b1, g1, r1, _ = cv2.split(img1)
    b2, g2, r2, alpha = cv2.split(img2)
    alpha = alpha.astype(float) / 255
    b = b2 * alpha + b1 * (1 - alpha)
    g = g2 * alpha + g1 * (1 - alpha)
    r = r2 * alpha + r1 * (1 - alpha)
    alpha = np.ones(alpha.shape, dtype=np.uint8) * 255
    return cv2.merge(
        (b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8), alpha))


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


def create_average_rasterized_image(img, cells, cell_size):
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
    for i in range(1, rows - 1):
        for j in range(1, columns - 1):
            classified_cells[i][j] = __getRisk(cells[i][j])
    return classified_cells


def __getRisk(cell):
    distances_and_risks = []
    for color, risk in COLOR_TO_RISK.items():
        distance = sum(pow(color[i] - cell[i], 2) for i in range(3))
        distances_and_risks.append((distance, risk))
    distances_and_risks.sort()
    return distances_and_risks[0][1]


def create_risk_rasterized_image(img, cells, cell_size):
    rows, columns = cells.shape
    height, width = rows * cell_size, columns * cell_size
    result = np.zeros((height, width, 4), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            for ii in range(cell_size):
                for jj in range(cell_size):
                    color = RISK_TO_COLOR[Risk(cells[i][j])]
                    result[i*cell_size + ii][j*cell_size + jj] = color
    return result


def sliceImage(img, cell_size):
    height, width, channels = img.shape
    i = 0
    while i < height:
        cv2.line(img, (0, i), (width, i), (0, 0, 0), 1)
        i += cell_size
    i = 0
    while i < width:
        cv2.line(img, (i, 0), (i, height), (0, 0, 0), 1)
        i += cell_size


def show_step(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)


def main():
    args = parse_args()

    print('Reading {} ...'.format(args.map))
    img = read_image_with_alpha(args.map)
    show_step(img)

    print('Adding grid ...')
    height, width, _ = img.shape
    grid = create_grid(height, width, args.cell_size)
    show_step(blend(img, grid))

    print('Averaging cells ...')
    cells = average_cells(img, args.cell_size)
    img = create_average_rasterized_image(img, cells, args.cell_size)
    height, width, _ = img.shape
    grid = create_grid(height, width, args.cell_size)
    show_step(blend(img, grid))

    print('Classifying cells ...')
    cells = classify_cells(cells)
    img = create_risk_rasterized_image(img, cells, args.cell_size)
    show_step(blend(img, grid))

    print('Eroding cells ...')
    #cells = 

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
