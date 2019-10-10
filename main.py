#!/usr/bin/env python

from enum import Enum
from math import pow

import numpy as np
import cv2


FILE = 'maps/map1.png'
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


def readImageWithAlpha(file_):
    img = cv2.imread(file_, cv2.IMREAD_UNCHANGED)
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))


def createGrid(height, width):
    grid = np.zeros((height, width, 4), dtype=np.uint8)
    for i in range(0, height, CELL_SIZE):
        cv2.line(grid, (0, i), (width, i), (0, 0, 0, 255), 1)
    for i in range(0, width, CELL_SIZE):
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


def averageCells(img):
    height, width, channels = img.shape
    rows = height // CELL_SIZE
    columns = width // CELL_SIZE
    cells = np.zeros((rows, columns, channels), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            irdi_sum = np.zeros(channels, dtype=int)
            for ii in range(CELL_SIZE):
                for jj in range(CELL_SIZE):
                    irdi_sum += img[i*CELL_SIZE + ii][j*CELL_SIZE + jj]
            cells[i][j] = irdi_sum // CELL_SIZE**2
    return cells


def createAverageRasterizedImage(img, cells):
    rows, columns, channels = cells.shape
    height = columns * CELL_SIZE
    width = rows * CELL_SIZE
    result = np.zeros((height, width, channels), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            for ii in range(CELL_SIZE):
                for jj in range(CELL_SIZE):
                    result[i*CELL_SIZE + ii][j*CELL_SIZE + jj] = cells[i][j]
    return result


def classifyCells(cells):
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


def createRiskRasterizedImage(img, cells):
    rows, columns = cells.shape
    height, width = rows * CELL_SIZE, columns * CELL_SIZE
    result = np.zeros((height, width, 4), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            for ii in range(CELL_SIZE):
                for jj in range(CELL_SIZE):
                    color = RISK_TO_COLOR[Risk(cells[i][j])]
                    result[i*CELL_SIZE + ii][j*CELL_SIZE + jj] = color
    return result


def sliceImage(img):
    height, width, channels = img.shape
    i = 0
    while i < height:
        cv2.line(img, (0, i), (width, i), (0, 0, 0), 1)
        i += CELL_SIZE
    i = 0
    while i < width:
        cv2.line(img, (i, 0), (i, height), (0, 0, 0), 1)
        i += CELL_SIZE


def showStep(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)


def main():
    print('Reading {} ...'.format(FILE))
    img = readImageWithAlpha(FILE)
    showStep(img)

    print('Adding grid ...')
    height, width, _ = img.shape
    grid = createGrid(height, width)
    showStep(blend(img, grid))

    print('Averaging cells ...')
    cells = averageCells(img)
    img = createAverageRasterizedImage(img, cells)
    height, width, _ = img.shape
    grid = createGrid(height, width)
    showStep(blend(img, grid))

    print('Classifying cells ...')
    cells = classifyCells(cells)
    img = createRiskRasterizedImage(img, cells)
    showStep(blend(img, grid))

    print('Eroding cells ...')
    #cells = 

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
