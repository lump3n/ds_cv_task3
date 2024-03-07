import cv2
import numpy as np


def print_image_info(image: np.ndarray) -> None:
    """
    Функция отображения информации об изображении в консоль

    Parameters
    ----------
    image : np.ndarray
        Изображение в виде матрицы чисел
    Returns
    -------
    None
    """
    image_height = image.shape[0]
    image_width = image.shape[1]
    number_of_channels = image.shape[-1] if image.ndim == 3 else 1

    print("Матрица изображения: \n", image[0])
    print(f"Высота: {image_height}", f"Ширина: {image_width}")
    print(f"Количество каналов: {number_of_channels}")
    print("- - - " * 10)


def count_center_pixel(image: np.ndarray) -> tuple[int, int]:
    """
    Функция рассчета центра изображения

    Parameters
    ----------
    image : np.ndarray
        Изображение в виде матрицы чисел

    Returns
    -------
    (int, int)
        Координаты центра изображения
    """
    height, width = image.shape[:2]
    center = (int(width / 2), int(height / 2))
    return center


def rotate_image(image: np.ndarray, degree: int) -> np.ndarray:
    """
    Функция поворота изображения на определенное количество градусов

    Parameters
    ----------
    image : np.ndarray
        Изображение в виде матрицы чисел

    degree : int
        Градусы поворота изображения.При положительном значении градусов(degree) поворот ПРОТИВ часовой стрелки,
        при отрицательном - ПО часовой стрелке,
    Returns
    -------
    np.ndarray
        Повернутое изображение
    """
    height, width = image.shape[:2]
    center = count_center_pixel(image)
    scale = 0.7

    matrix = cv2.getRotationMatrix2D(center, degree, scale)
    rotated_image = cv2.warpAffine(image, matrix, (width, height))
    return rotated_image


def save_image(filename: str, image: np.ndarray, file_path: str) -> None:
    """
    Функция сохранения изображения

    Parameters
    ----------
    filename : str
        Имя сохраняемого изображения
    image : np.ndarray
        Изображение в виде матрицы чисел
    file_path : str
        Путь сохранения изображения
     Returns
    -------
    None
    """
    cv2.imwrite(file_path + filename + '.png', image)


def show_image(image: np.ndarray, image_title: str) -> None:
    """
    Функция отображения изображения

    Parameters
    ----------
    image : np.ndarray
        Изображение в виде матрицы чисел
    image_title: str
        Название изображение
    Returns
    -------
    None
    """
    cv2.namedWindow('PRESS ANY KEY TO CONTINUE - ' + image_title, cv2.WINDOW_NORMAL)
    cv2.imshow('PRESS ANY KEY TO CONTINUE - ' + image_title, image)
    cv2.waitKey(0)
    cv2.destroyWindow('PRESS ANY KEY TO CONTINUE - ' + image_title)


def mirror_image(image: np.ndarray, mirror_flag: str = 'vertical') -> np.ndarray:
    """
    Функция отражения изображения

    Parameters
    ----------
    image : np.ndarray
        Изображение в виде матрицы чисел
    mirror_flag : str
        Выбор направления отражения:
        vertical - вертикальное отражени,
        horizontal - горизонтальное отражение,
        both - горизонтальое и вертикальное отражение одновременно.

    Returns
    -------
    np.ndarray
        Отраженное изображение
    """
    mirror_flags = {'vertical': 1,
                    'horizontal': 0,
                    'both': -1}
    mirrored_image = cv2.flip(image, mirror_flags[mirror_flag])
    return mirrored_image


COLOR_BLUE = (0, 0, 255)
COLOR_RED = (255, 0, 0)
COLOR_YELLOW = (255, 255, 0)
PATH_SAVE_IMAGE = "data/save/"
TEXT_FONT = cv2.FONT_HERSHEY_COMPLEX_SMALL


def main():
    # Загрузка изображение и отображение его на экране
    image = cv2.imread("data/images/pcb.jpg")
    show_image(image, 'Изначальное изображение')

    # Вывод в консоль информации об изображении
    print('ИНФОРМАЦИЯ ИЗНАЧАЛЬНОГО ИЗОБРАЖЕНИЯ')
    print_image_info(image)

    # Изменение разрешения изображения и отображение информации о нем в консоли
    resized_image = cv2.resize(image, None, fx=0.75, fy=0.75)
    print('ИНФОРМАЦИЯ ИЗОБРАЖЕНИЯ C ИЗМЕНЕННЫМ РАЗРЕШЕНИЕМ')
    print_image_info(resized_image)

    # Повернуть изображение на 45, 90 и 180 градусов
    image_45_degree_rotate = rotate_image(image=image, degree=45)
    show_image(image_45_degree_rotate, 'Изображение повернутое на 45 градусов')
    save_image(filename='image_45_degree_rotate', image=image_45_degree_rotate, file_path=PATH_SAVE_IMAGE)

    image_90_degree_rotate = rotate_image(image=image, degree=90)
    show_image(image_90_degree_rotate, 'Изображение повернутое на 90 градусов')
    save_image(filename='image_90_degree_rotate', image=image_90_degree_rotate, file_path=PATH_SAVE_IMAGE)

    image_180_degree_rotate = rotate_image(image=image, degree=180)
    show_image(image_180_degree_rotate, 'Изображение повернутое на 180 градусов')
    save_image(filename='image_180_degree_rotate', image=image_180_degree_rotate, file_path=PATH_SAVE_IMAGE)

    # Отражение изображения по вертикали и горизонтали
    image_mirrored_by_vertical = mirror_image(image, 'vertical')
    show_image(image_mirrored_by_vertical, 'Изображение отраженное по вертикали')
    save_image(filename='image_mirrored_by_vertical', image=image_mirrored_by_vertical,
               file_path=PATH_SAVE_IMAGE)

    image_mirrored_by_horizontal = mirror_image(image, 'horizontal')
    show_image(image_mirrored_by_horizontal, 'Изображение отраженное по горизонтали')
    save_image(filename='image_mirrored_by_horizontal', image=image_mirrored_by_horizontal,
               file_path=PATH_SAVE_IMAGE)

    # Обрезание области размером 100x100 пикселей
    cropped_image = image[100:200, 100:200]
    show_image(cropped_image, 'Обрезанная область изображения 100х100')

    # Вывод в консоль значения центрального пикселя изображения
    # Измение значения данного пикселя на (0, 0, 255)
    center = count_center_pixel(cropped_image)
    print('Центральный пиксель обрезанной области изображения: ', center)
    print("- - - " * 10)
    cropped_image[center] = COLOR_BLUE
    show_image(cropped_image, 'Замена центрального пикселя обрезанного изображения синим цветом')

    # Измение значений группы пикселей на красный цвет
    cropped_image[20:30, 20:30] = COLOR_RED
    show_image(cropped_image, 'Замена группы пикселей обрезанного изображения красным цветом')

    # Отрисовка прямоугольника вокруг области измененных пикселей
    cropped_image = cv2.rectangle(cropped_image, (18, 18), (31, 31), COLOR_YELLOW, 1)
    show_image(cropped_image, 'Отрисовка прямоугольника вокруг области измененных пикселей')

    # Добавление текста rect
    cropped_image = cv2.putText(cropped_image, "rect", (32, 16), TEXT_FONT, 0.7, COLOR_YELLOW, 1)
    show_image(cropped_image, 'Добавление текста rect')
    save_image(filename='cropped_image', image=cropped_image,
               file_path=PATH_SAVE_IMAGE)

    # Дополнительно
    # Проверка исправления функции print_image_info
    print('ИНФОРМАЦИЯ ИЗНАЧАЛЬНОГО ИЗОБРАЖЕНИЯ')
    print_image_info(image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print('ИНФОРМАЦИЯ ИЗОБРАЖЕНИЯ В ПРОСТРАСНТВЕ ЦВЕТОВ GRAY')
    print_image_info(image_gray)


if __name__ == '__main__':
    main()
