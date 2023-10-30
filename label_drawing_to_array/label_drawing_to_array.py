from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt

from label_drawing_to_array.objects_drawing_and_array import ObjectsDrawingAndArray
from label_drawing_to_array.drawer_for_label import DrawingLabel

import numpy as np


# Свой лейбл для рисования и перевода в распределение
class LabelDrawingToArray(QtWidgets.QLabel):

    def __init__(self):
        # Инициализируем лейбл
        super().__init__()

        # Создаем pixmap и закрашиваем в белый
        pixmap = QtGui.QPixmap(600, 300)
        pixmap.fill(Qt.white)

        # Масштабируем под размер выделенной области и отображаем (устанавливаем)
        pixmap.scaled(self.width(), self.height())
        self.setPixmap(pixmap)

        # Параметры для рисования. Последние координаты и цвет ручки
        self.last_x, self.last_y = None, None
        self.pen_color = QtGui.QColor(Qt.blue)

        # Координаты сетки
        self.coordinates_grids_x = np.array([])
        self.coordinates_grids_y = np.array([])

        self.objects_drawing_and_array = None

    # При изменении размеров элемента очистка и перерисовка
    def resizeEvent(self, event):
        self.cleaning_and_mesh()

    # Методы рисования
    def mouseMoveEvent(self, e):
        if self.last_x is None:  # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return  # Ignore the first time.

        painter = QtGui.QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(3)
        p.setColor(self.pen_color)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    # Рисование сетки и подписей координат
    def cleaning_and_mesh(self):
        # (0) Получаем объекты для рисования
        # Пиксельная карта на которой рисуем
        pixmap = QtGui.QPixmap(self.width(), self.height())
        pixmap.fill(Qt.white)
        # Кисть рисования на холсте - pixmap
        painter = QtGui.QPainter(pixmap)

        self.objects_drawing_and_array = ObjectsDrawingAndArray(pixmap=pixmap,
                                                                painter=painter)

        # (1) Рисуем сетку
        self.coordinates_grids_x, self.coordinates_grids_y = \
            DrawingLabel.drawing_field(self.objects_drawing_and_array)

        # (2) Подписываем оси
        DrawingLabel.drawing_axis_labels(self.objects_drawing_and_array,
                                         self.coordinates_grids_x, self.coordinates_grids_y)

        painter.end()
        self.setPixmap(pixmap)

    # Метод предоставляющий данные полученные из графика в виде массива размера 256
    def graph_in_distribution(self):
        # Проверка отрисовки сетки
        if self.objects_drawing_and_array is None:
            return

        # Обновляем pixmap, для получения рисунка с мышью
        self.objects_drawing_and_array.pixmap = self.pixmap()
        # Переводим рисунок в массив
        line_coordinate_array = self.objects_drawing_and_array.graph_in_distribution(self.pen_color)

        # Данные не пустые
        if not line_coordinate_array:
            return

        # Массив произвольной длинны приводим к 256
        mass = np.zeros(256, dtype="int64")
        number_elements_subinterval = len(line_coordinate_array) / 256

        for i in range(256):
            start_subinterval = int(i * number_elements_subinterval)
            end_subinterval = int((i + 1) * number_elements_subinterval)

            interval = line_coordinate_array[start_subinterval: end_subinterval]

            mass[i] = int(sum(interval) / len(interval))

        return mass
