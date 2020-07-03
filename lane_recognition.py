import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from pathlib import Path


INPUT_FOLDER = Path("./images_clip")
OUTPUT_FOLDER = Path("./images_processed")
DEBUG = True


def plot_image(image, title):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(title)
    plt.show()


def save_image(image, title):
    cv2.imwrite(title, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def enhance_blacks(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if DEBUG:
        plot_image(img2, f"enhanced blacks image")
    return img2


def grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if DEBUG:
        plot_image(grayscale_image, "grayscale")
    return grayscale_image


def enhance_contrast(image):
    alpha = 1.0
    beta = 10
    enhanced_contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    if DEBUG:
        plot_image(enhanced_contrast_image, f"enhanced contrast image, beta: {beta}")
    return enhanced_contrast_image


def blur(image):
    blur_image = cv2.GaussianBlur(image, (7, 7), 0)
    if DEBUG:
        plot_image(blur_image, "blur")
    return blur_image


def sobel_operator(img):
    gray_img = img
    h, w = gray_img.shape
    # define filters
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
    vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1
    # define images with 0s
    newhorizontalImage = np.zeros((h, w))
    newverticalImage = np.zeros((h, w))
    newgradientImage = np.zeros((h, w))

    # offset by 1
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            horizontalGrad = (
                (horizontal[0, 0] * gray_img[i - 1, j - 1])
                + (horizontal[0, 1] * gray_img[i - 1, j])
                + (horizontal[0, 2] * gray_img[i - 1, j + 1])
                + (horizontal[1, 0] * gray_img[i, j - 1])
                + (horizontal[1, 1] * gray_img[i, j])
                + (horizontal[1, 2] * gray_img[i, j + 1])
                + (horizontal[2, 0] * gray_img[i + 1, j - 1])
                + (horizontal[2, 1] * gray_img[i + 1, j])
                + (horizontal[2, 2] * gray_img[i + 1, j + 1])
            )

            newhorizontalImage[i - 1, j - 1] = abs(horizontalGrad)

            verticalGrad = (
                (vertical[0, 0] * gray_img[i - 1, j - 1])
                + (vertical[0, 1] * gray_img[i - 1, j])
                + (vertical[0, 2] * gray_img[i - 1, j + 1])
                + (vertical[1, 0] * gray_img[i, j - 1])
                + (vertical[1, 1] * gray_img[i, j])
                + (vertical[1, 2] * gray_img[i, j + 1])
                + (vertical[2, 0] * gray_img[i + 1, j - 1])
                + (vertical[2, 1] * gray_img[i + 1, j])
                + (vertical[2, 2] * gray_img[i + 1, j + 1])
            )

            newverticalImage[i - 1, j - 1] = abs(verticalGrad)

            # Edge Magnitude
            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            newgradientImage[i - 1, j - 1] = mag
    return newgradientImage


def find_street_lanes(image):
    enhanced_blacks_image = enhance_blacks(image)
    grayscale_image = grayscale(enhanced_blacks_image)
    enhanced_contrast_image = enhance_contrast(grayscale_image)
    blur_image = blur(enhanced_contrast_image)
    canny_image = sobel_operator(blur_image)
    roi_image = roi(canny_image)
    hough_lines_image = hough_lines(roi_image, 1, np.pi / 180, 100, 20, 50)
    final_image = combine_images(hough_lines_image, image)
    return final_image


def roi(image):
    # Excluding region that is not the actual road; case-specific
    bottom_padding = 40  # Front bumper compensation
    height = image.shape[0]
    width = image.shape[1]
    bottom_left = [100, height - bottom_padding]
    bottom_right = [width, height - bottom_padding]
    top_right = [width * 1 / 3, height * 2 / 3]
    top_left = [width * 2 / 3, height * 2 / 3]
    vertices = [
        np.array([bottom_left, bottom_right, top_left, top_right], dtype=np.int32)
    ]
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    if DEBUG:
        plot_image(mask, "mask")
    masked_image = cv2.bitwise_and(image, mask)
    if DEBUG:
        plot_image(masked_image, "roi")
    (thresh, binary_image) = cv2.threshold(masked_image, 127, 255, cv2.THRESH_BINARY)
    if DEBUG:
        plot_image(binary_image, "ROI binary")
    save_image(binary_image, "roi_binary.jpg")
    binary_image = cv2.imread("roi_binary.jpg", 0)
    os.remove("roi_binary.jpg")
    return binary_image


def averaged_lines(image, lines):
    def merge_lines(image, lines):
        if len(lines) > 0:
            slope, intercept = np.average(lines, axis=0)
            y1 = image.shape[0]
            y2 = int(y1 * (17 / 24))
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            return np.array([x1, y1, x2, y2])

    right_lines = []
    left_lines = []
    outer_lines = []
    for x1, y1, x2, y2 in lines[:, 0]:
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        # Different values of slopes allow identifying lanes
        if 0.4 < slope:
            right_lines.append([slope, intercept])
        # CAVEAT: -0.2 catches the outer lane as well
        elif -0.24 > slope > -0.4:
            outer_lines.append([slope, intercept])
        elif -0.85 > slope:
            left_lines.append([slope, intercept])

    lines_to_draw = []
    left = merge_lines(image, left_lines)
    right = merge_lines(image, right_lines)
    lines_to_draw.append(left)
    lines_to_draw.append(right)
    if outer_lines:
        outer = merge_lines(image, outer_lines)
        lines_to_draw.append(outer)
    return tuple(lines_to_draw)


def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    lines = cv2.HoughLinesP(
        image,
        rho,
        theta,
        threshold,
        np.array([]),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    if lines is not None:
        lines = averaged_lines(image, lines)
        for counter, line in enumerate(lines):
            if line is not None:
                x1, y1, x2, y2 = line
                if counter == 2:
                    cv2.line(lines_image, (x1, y1), (x2, y2), (180, 0, 0), 10)
                else:
                    cv2.line(lines_image, (x1, y1), (x2, y2), (0, 180, 0), 10)
        if DEBUG:
            plot_image(lines_image, "lines")
    return lines_image


def combine_images(image, initial_image, α=0.9, β=1.0, λ=0.0):
    combined_image = cv2.addWeighted(initial_image, α, image, β, λ)
    if DEBUG:
        plot_image(combined_image, "combined")
    return combined_image


def sorter(filename):
    key = int("".join([x for x in filename.split(".")[0] if x.isdigit()]))
    return key


# Finding images in the INPUT_FOLDER and sorting them
images = [x for x in os.listdir(INPUT_FOLDER) if x.split(".")[-1] == "jpg"]
images = sorted(images, key=sorter)

# The actual script
for counter, image_path in enumerate(images, 1):
    image = cv2.imread(str(INPUT_FOLDER) + "/" + image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plot_image(image, "test")
    street_lanes = find_street_lanes(image)
    street_lanes = cv2.cvtColor(street_lanes, cv2.COLOR_RGB2BGR)
    save_image(street_lanes, str(OUTPUT_FOLDER) + f"/image_processed_{counter}.jpg")
