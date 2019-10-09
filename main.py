#!/usr/bin/env python

import numpy as np
import cv2

FILE = 'maps/mapa_irdi_2019_10_08.jpg'
CELL_SIZE = 200


def calculateCells(img):
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


def calculateMaxNeighbours(cells):
    rows, columns, channels = cells.shape
    max_neighbours = np.zeros((rows, columns, 2), dtype=np.int8)
    for i in range(1, rows - 1):
        for j in range(1, columns - 1):
            max_neighbour = np.array((0, 0, 255))
            max_ii, max_jj = 0, 0
            for ii in range(-1, 1):
                for jj in range(-1, 1):
                    cell = cells[i + ii][ j + jj]
                    if __greaterThan(cell, max_neighbour):
                        max_neighbour = cell
                        max_ii, max_jj = ii, jj
            print(max_neighbour)
            max_neighbours[i][j] = (max_ii, max_jj)
    return max_neighbours


def __greaterThan(color1, color2):
    """
    Blue:   (0,     0, 255)
    Green:  (0,   255,   0)
    Yellow: (255, 255,   0)
    Orange: (255, 127,   0)
    Red :   (255,   0,   0)
    """
    if color1[0] > color2[0]:
        return True
    if color1[0] == color2[0]:
        if color1[1] < color2[1]:
            return True
        if color1[1] == color2[1] and color1[2] < color2[2]:
            return True
    return False


def createRasterizedImage(img, cells):
    rows, columns, _ = cells.shape
    height, width = rows * CELL_SIZE, columns * CELL_SIZE
    result = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
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
    print('Calculating cells ...')
    cells = calculateCells(img)
    print(cells)
    print('Calculating max neighbours ...')
    max_neighbours = calculateMaxNeighbours(cells)
    print(max_neighbours)
    print('Rasterizing ...')
    img = createRasterizedImage(img, cells)
    print('Slicing ...')
    sliceImage(img)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
