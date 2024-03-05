import cv2
import numpy as np
from introduction_to_image_operations import show_image


def make_images_in_different_color_spaces(image):
    color_spaces = ('RGB', 'XYZ', 'HSV', 'HLS', 'LAB', 'LUV', 'YUV', 'GRAY')
    color_images = {color: cv2.cvtColor(image, getattr(cv2, 'COLOR_BGR2' + color))
                    for color in color_spaces}
    color_images['BGR'] = image
    color_images['CMYK'] = bgr2cmyk(image)
    return color_images


def bgr2cmyk(image):
    image = image.astype(float) / 255.
    k = 1 - np.max(image, axis=2)
    c = (1 - image[..., 2] - k) / (1 - k)
    m = (1 - image[..., 1] - k) / (1 - k)
    y = (1 - image[..., 0] - k) / (1 - k)
    image_cmyk = (np.dstack((c, m, y, k)) * 255).astype(np.uint8)
    return image_cmyk


def main():
    # Загрузка изображение и отображение его на экране
    image = cv2.imread("data/images/pcb.jpg")
    show_image(image, 'Изначальное изображение')

    # Перевод изображения в различные цветовые пространства
    color_space_images = make_images_in_different_color_spaces(image)
    for space in color_space_images:
        show_image(image=color_space_images[space], image_title=space)

    # Разделение изображение на каналы
    b, g, r = cv2.split(image)
    show_image(b, 'Синий канал')
    show_image(g, 'Зеленый канал')
    show_image(r, 'Красный канал')

    # Обработка одного из каналов изображения с помощью базовых мат. операций
    b, g, r = cv2.split(color_space_images['BGR'])
    show_image(r, 'Красный канал до обработки')

    r = r * 4
    show_image(r, 'Увеличение значений красного канаала в 4 раза')

    r = r + 5
    show_image(r, 'Увеличение значений красного канаала на 5')

    r = (r / 2)
    r = cv2.normalize(r, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    show_image(r, 'Уменьшение значений красного канаала в 2 раза')

    r = r - 2
    show_image(r, 'Уменьшение значений красного канаала на 2')

    # Объединение каналов разделенного изображения
    merged_image = cv2.merge((b, g, r))
    show_image(merged_image, 'Изображение из объединенных каналов')

    # Тестирование низкочастотных фильтров
    image_blur = cv2.blur(image, (20, 20))
    show_image(image_blur, 'Изображение обработанное функцией blur')

    image_gaussian_blur = cv2.GaussianBlur(image, (19, 19), 0)
    show_image(image_gaussian_blur, 'Изображение обработанное функцией GaussianBlur')

    image_median_blur = cv2.medianBlur(image, 19)
    show_image(image_median_blur, 'Изображение обработанное функцией medianBlur')

    image_bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)
    show_image(image_bilateral_filter, 'Изображение обработанное функцией bilateralFilter')

    # Тестирование высокочастотных фильтров
    image_canny = cv2.Canny(image, 50, 150)
    show_image(image_canny, 'Изображение обработанное функцией Canny')

    image_laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
    image_laplacian = cv2.convertScaleAbs(image_laplacian)
    show_image(image_laplacian, 'Изображение обработанное функцией Laplacian')

    grad_sobel_x = cv2.convertScaleAbs(cv2.Sobel(image, cv2.CV_16S, 1, 0))
    grad_sobel_y = cv2.convertScaleAbs(cv2.Sobel(image, cv2.CV_16S, 0, 1))
    image_sobel = cv2.addWeighted(grad_sobel_x, 0.5, grad_sobel_y, 0.5, 0)
    show_image(image_sobel, 'Изображение обработанное функцией Sobel')

    grad_scharr_x = cv2.convertScaleAbs(cv2.Scharr(image, cv2.CV_16S, 1, 0))
    grad_scharr_y = cv2.convertScaleAbs(cv2.Scharr(image, cv2.CV_16S, 0, 1))
    image_scharr = cv2.addWeighted(grad_scharr_x, 0.5, grad_scharr_y, 0.5, 0)
    show_image(image_scharr, 'Изображение обработанное функцией Scharr')

    # Поиск границ золотой подложки
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(image)
    v += 255
    image = cv2.merge((h, s, v))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.medianBlur(image, 19)
    image_x = cv2.convertScaleAbs(cv2.Scharr(image, cv2.CV_16S, 1, 0))
    image_y = cv2.convertScaleAbs(cv2.Scharr(image, cv2.CV_16S, 0, 1))
    image = cv2.addWeighted(image_x, 0.5, image_y, 0.5, 0)
    show_image(image, 'Поиск границ золотой подложки')


if __name__ == '__main__':
    main()
