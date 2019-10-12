#!/usr/bin/env python

import argparse
from enum import Enum
from math import pow, cos, sin

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


def show_step(img, delay):
    cv2.imshow('image', img)
    cv2.waitKey(delay)


def create_grid(height, width, cell_size):
    grid = np.zeros((height, width, 4), dtype=np.uint8)
    for i in range(0, height, cell_size):
        cv2.line(grid, (0, i), (width, i), (0, 0, 0, 255), 1)
    for i in range(0, width, cell_size):
        cv2.line(grid, (i, 0), (i, height), (0, 0, 0, 255), 1)
    return grid


def blend(img1, img2):
    b1, g1, r1, alpha1 = cv2.split(img1)
    b2, g2, r2, alpha2 = cv2.split(img2)
    alpha = alpha2.astype(float) / 255
    b = b2 * alpha + b1 * (1 - alpha)
    g = g2 * alpha + g1 * (1 - alpha)
    r = r2 * alpha + r1 * (1 - alpha)
    alpha = (alpha1 + alpha2) // 2
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
            classified_cells[i][j] = __getRisk(cells[i][j])
    return classified_cells


def __getRisk(cell):
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


def calculate_forces_on_borders(cells):
    rows, columns = cells.shape
    forces = np.zeros((rows, columns, 2), dtype=float)
    row_steps = (-1, -1, -1,  0, 0,  1, 1, 1);
    col_steps = (-1,  0,  1, -1, 1, -1, 0, 1);
    for i in range(rows):
        for j in range(columns):
            risk = cells[i][j]
            neighbours = {}
            for ii, jj in zip(row_steps, col_steps):
                row = (i + ii) % rows
                column = (j + jj) % columns
                neighbour_risk = cells[row][column]
                if neighbour_risk in neighbours:
                    neighbours[neighbour_risk].append(
                        np.array((jj, ii)))
                else:
                    neighbours[neighbour_risk] = [np.array((jj, ii))]
            forces[i][j] = __get_gradient_force(risk, neighbours)
    return forces


def __get_gradient_force(risk, neighbours):
    """
    Calculates the vector that points to the maximum risk.
    """
    if (len(neighbours) == 1 and risk in neighbours):
        # There is no gradient, all cells are in the same region.
        return np.zeros(2)

    highest_risk = max(neighbours.keys())
    if highest_risk < risk:
        # Only touching lower risk cells.
        return np.zeros(2)

    gradient_force = np.zeros(2)
    for force in neighbours[highest_risk]:
        gradient_force += force
    magnitude = np.linalg.norm(gradient_force)
    if magnitude:
        return gradient_force / magnitude
    return np.zeros(2)


def create_forces_image(forces, cell_size):
    rows, columns, _ = forces.shape
    img = np.zeros((rows * cell_size, columns * cell_size, 4), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            force = forces[i][j]
            if np.count_nonzero(force):
                force = force * 1/2 * cell_size
                center_x = j * cell_size + cell_size / 2
                center_y = i * cell_size + cell_size / 2
                center = np.array((center_x, center_y), dtype=int)
                point1 = center - 1/2 * force
                point2 = center + 1/2 * force
                point1 = tuple(point1.astype(int))
                point2 = tuple(point2.astype(int))
                cv2.arrowedLine(
                    img, point1, point2, (0, 0, 0, 255), 1,
                    line_type=cv2.LINE_8, tipLength=.25)
    return img


def grow_field_of_forces(forces, min_influence=3):
    grown_forces = np.zeros(forces.shape)
    rows, columns, _ = forces.shape
    row_steps = (-1, -1, -1,  0, 0,  1, 1, 1);
    col_steps = (-1,  0,  1, -1, 1, -1, 0, 1);
    for i in range(rows):
        for j in range(columns):
            force = forces[i][j]
            if np.count_nonzero(force):
                grown_forces[i][j] = force
            else:
                neighbours = []
                for ii, jj in zip(row_steps, col_steps):
                    row = (i + ii) % rows
                    column = (j + jj) % columns
                    neighbour_force = forces[row][column]
                    if np.count_nonzero(neighbour_force):
                        neighbours.append(neighbour_force)
                if len(neighbours) >= min_influence:
                    grown_forces[i][j] = __getMeanForce(neighbours)
    return grown_forces


def __getMeanForce(forces):
    mean_force = np.zeros(2)
    for force in forces:
        mean_force += force
    if np.count_nonzero(mean_force):
        return mean_force / np.linalg.norm(mean_force)
    return np.zeros(2)


def all_forces_are_non_zero(forces):
    rows, columns, _ = forces.shape
    for i in range(rows):
        for j in range(columns):
            if not np.count_nonzero(forces[i][j]):
                return False
    return True


def main():
    args = parse_args()

    print('Reading {} ...'.format(args.map))
    img = read_image_with_alpha(args.map)
    show_step(img, 0)

    print('Adding grid ...')
    height, width, _ = img.shape
    grid = create_grid(height, width, args.cell_size)
    show_step(blend(img, grid), 0)

    print('Averaging cells ...')
    avg_cells = average_cells(img, args.cell_size)
    img = create_rasterized_image(avg_cells, args.cell_size)
    height, width, _ = img.shape
    grid = create_grid(height, width, args.cell_size)
    show_step(blend(img, grid), 0)

    print('Classifying cells ...')
    risk_cells = classify_cells(avg_cells)
    color_cells = risk_cells_to_color_cells(risk_cells)
    img = create_rasterized_image(color_cells, args.cell_size)
    show_step(blend(img, grid), 0)

    print('Smoothing cells ...')
    for i in range(20):
        print(' - iteration ', i)
        smoothed_risk_cells = smooth(risk_cells, 1)
        color_cells = risk_cells_to_color_cells(smoothed_risk_cells)
        img = create_rasterized_image(color_cells, args.cell_size)
        show_step(blend(img, grid), 1)
        if np.array_equal(smoothed_risk_cells, risk_cells):
            break
        risk_cells = smoothed_risk_cells
    show_step(blend(img, grid), 0)

    print('Calculating forces on borders ...')
    forces = calculate_forces_on_borders(risk_cells)
    forces_img = create_forces_image(forces, args.cell_size)
    show_step(blend(img, blend(grid, forces_img)), 0)

    print('Growing field of forces ...')
    rows, columns, _ = forces.shape
    forces_count = rows * columns
    for i in range(100):
        print(' - iteration ', i)
        forces = grow_field_of_forces(forces)
        forces_img = create_forces_image(forces, args.cell_size)
        show_step(blend(img, blend(grid, forces_img)), 1)
        if all_forces_are_non_zero(forces):
            break
    show_step(blend(img, blend(grid, forces_img)), 0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
