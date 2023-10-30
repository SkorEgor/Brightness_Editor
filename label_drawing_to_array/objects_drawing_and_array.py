from PyQt5 import QtGui
from PyQt5.QtCore import Qt


# Объекты / параметры рисунка и метод перевода рисунка в массив
class ObjectsDrawingAndArray:
    def __init__(self, pixmap, painter):
        # Массив 256 * 256
        size_distribution = 256

        # Параметры поля
        self.cells_width = 15
        self.cells_height = 15
        self.maximum_radius_value_x = size_distribution - 1
        self.maximum_radius_value_y = size_distribution - 1

        # Объекты для рисования
        self.pixmap = pixmap
        self.painter = painter

    def graph_in_distribution(self, pen_color):
        # Проверяем что данные есть
        if self.pixmap is None:
            return

        # Для получения пикселей конвертируем
        image = self.pixmap.toImage()

        # Размеры изображения
        image_size = image.size()
        width, height = image_size.width(), image_size.height()

        # Цвет ручки в rgb
        pen_color_rgb = pen_color.getRgb()

        # Поиск границ сетки
        x_start = 0
        while QtGui.QColor((image.pixel(x_start, height // 2))).getRgb() != QtGui.QColor(
                Qt.black).getRgb() and x_start < width:
            x_start += 1

        x_end = width - 1
        while QtGui.QColor((image.pixel(x_end, height // 2))).getRgb() != QtGui.QColor(
                Qt.black).getRgb() and x_end >= 0:
            x_end -= 1

        y_start = 0
        while QtGui.QColor((image.pixel(width // 2, y_start))).getRgb() != QtGui.QColor(
                Qt.black).getRgb() and y_start < height:
            y_start += 1

        y_end = width - 1
        while QtGui.QColor((image.pixel(width // 2, y_end))).getRgb() != QtGui.QColor(
                Qt.black).getRgb() and y_end >= 0:
            y_end -= 1

        # Проходим в диапазоне сетки
        # Поднимаясь пока не встретим синюю ручку
        list_val = [0] * (x_end - x_start)

        for x in range(x_start, x_end):
            for y in range(y_end, y_start, -1):

                if QtGui.QColor((image.pixel(x, y))).getRgb() == pen_color_rgb:
                    list_val[x - x_start] = y_end - y
                    break

        return list_val
