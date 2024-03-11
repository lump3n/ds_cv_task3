import cv2
from introduction_to_image_operations import show_image


def main():
    # Загрузка изображение и отображение его на экране
    image = cv2.imread("data/images/pcb.jpg")
    show_image(image, 'Изначальное изображение')

    # Удаление мелких частей
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_thresh = cv2.threshold(image_gray, 110, 255, cv2.THRESH_BINARY)
    image_blured = cv2.medianBlur(image_thresh, 23)
    image_mask = cv2.inRange(image_blured, 135, 255)
    show_image(image_mask, 'Маска изображения после пороговой обработки и блюра')

    # Выделение только внешних контуров деталей платы при помощи мода RETR_EXTERNAL
    contours, _ = cv2.findContours(image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отсеивание контуров по их размерам и отрисовка на изображении
    image_with_filled_contour = image.copy()
    image_with_bounding_rectangle = image.copy()
    for num, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # Если длина или ширина контура входит в данные границы, то данный контур не будет отрисовываться
        if w > 100 or h > 100 or w < 20 or h < 20:
            pass
        else:
            # Отрисовка контуров при помощи drawContours и fillPoly
            image_with_filled_contour = cv2.drawContours(image_with_filled_contour, [contours[num]], 0, (0, 0, 255), 5)
            cv2.fillPoly(image_with_filled_contour, pts=[contours[num]], color=(0, 0, 255))
            # Отрисовка контуров при помощи rectangle
            cv2.rectangle(image_with_bounding_rectangle, (x, y), (x + w, y + h), (0, 0, 250), 5)
    # Вывод итоговых изображений
    show_image(image_with_filled_contour, 'Итоговое изображение, сделанное при помощи функций drawContours и fillPoly')
    show_image(image_with_bounding_rectangle, 'Итоговое изображение, сделанное при помощи отрисовки ограничивающий '
                                              'прямоугольника для каждого контура')


if __name__ == '__main__':
    main()
