#!/usr/bin/env python

import numpy as np
import cv2

FILE = 'maps/mapa_irdi_2019_10_08.jpg'
CELL_SIZE = 10


def calculateCells(img):
    height, width, channels = img.shape
    cells = {}
    for i in range(height // CELL_SIZE):
        for j in range(width // CELL_SIZE):
            irdi_sum = np.zeros(channels, dtype=int)
            for ii in range(CELL_SIZE):
                for jj in range(CELL_SIZE):
                    irdi_sum += img[i*CELL_SIZE + ii][j*CELL_SIZE + jj]
            cells[(i, j)] = irdi_sum // CELL_SIZE**2
    return cells, i, j


def createRasterizedImage(img, cells, columns, rows):
    height = columns * CELL_SIZE
    width = rows * CELL_SIZE
    result = np.zeros((columns*CELL_SIZE, width, 3), dtype=np.uint8)
    for i in range(height // CELL_SIZE):
        for j in range(width // CELL_SIZE):
            for ii in range(CELL_SIZE):
                for jj in range(CELL_SIZE):
                    result[i*CELL_SIZE + ii][j*CELL_SIZE + jj] = cells[(i, j)]
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


def main():
    img = cv2.imread(FILE, cv2.IMREAD_UNCHANGED)
    cells, columns, rows = calculateCells(img)
    img = createRasterizedImage(img, cells, columns, rows)
    sliceImage(img)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
