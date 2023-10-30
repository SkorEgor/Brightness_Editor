import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import Qt

from label_drawing_to_array.objects_drawing_and_array import ObjectsDrawingAndArray


# Рисует на pixmap, готовые элементы
class DrawingLabel:
    # Отступы от границы элемента
    padding_x = 40
    padding_y = 20
    # Кисти
    pen_grid = QtGui.QPen(Qt.black, 2)
    pen_text = QtGui.QPen(Qt.red, 1)
    # Шрифты
    font_text = QtGui.QFont("Times", 12)

    # Рисование сетки
    @staticmethod
    def drawing_field(draw: ObjectsDrawingAndArray):
        # Выбираем кисть
        draw.painter.setPen(DrawingLabel.pen_grid)  # Устанавливаем цвети размер

        # Считаем расстояние между линиями сетки в пикселях
        step_width = (draw.pixmap.width() - DrawingLabel.padding_x * 2) / draw.cells_width
        step_height = (draw.pixmap.height() - DrawingLabel.padding_y * 2) / draw.cells_height

        # Границы области отрисовки
        # По x - ширине width
        left_border = DrawingLabel.padding_x
        right_border = int(draw.pixmap.width() - DrawingLabel.padding_x)
        # По y - ширине height
        top_border = DrawingLabel.padding_y
        bottom_border = int(draw.pixmap.height() - DrawingLabel.padding_y)

        # Для вертикальных линий - координаты по x
        coordinates_x = np.arange((draw.cells_width + 1), dtype="float16")
        coordinates_x = (DrawingLabel.padding_x + coordinates_x * step_width).astype("uint16")
        # Для горизонтальных линий - координаты по x
        coordinates_y = np.arange((draw.cells_height + 1), dtype="float16")
        coordinates_y = (DrawingLabel.padding_y + coordinates_y * step_height).astype("uint16")

        # Отрисовка линий
        # Вертикальные - смещены по x
        for x_i in coordinates_x:
            draw.painter.drawLine(x_i, top_border, x_i, bottom_border)
        # Горизонтальные - смещены по y
        for y_i in coordinates_y:
            draw.painter.drawLine(left_border, y_i, right_border, y_i)

        return coordinates_x, coordinates_y

    # Подпись делений осей сетки в pixmap
    @staticmethod
    def drawing_axis_labels(
            draw: ObjectsDrawingAndArray,
            coordinates_grids_x, coordinates_grids_y
    ):
        # Цена деления
        step_radius_x = int(draw.maximum_radius_value_x / draw.cells_width)
        step_radius_y = int(draw.maximum_radius_value_y / draw.cells_height)

        # Значения делений для координаты
        # По оси x
        val_calibration_x = np.arange(0, draw.cells_width + 1, dtype="uint16")
        val_calibration_x = val_calibration_x * step_radius_x
        # По оси y
        val_calibration_y = np.arange(draw.cells_height, - 1, -1, dtype="uint16")
        val_calibration_y = val_calibration_y * step_radius_y

        # Ручка для написания теста и параметры текста
        draw.painter.setPen(DrawingLabel.pen_text)
        draw.painter.setFont(DrawingLabel.font_text)

        # # Подписываем значения делений осй
        # # Для оси y`
        for line_i in range(draw.cells_height + 1):
            # Подпись значений
            draw.painter.drawText(0, coordinates_grids_y[line_i],
                                  f' {val_calibration_y[line_i]}')
        # Для оси x
        for column_i in range(draw.cells_width + 1):
            # Подпись значений
            draw.painter.drawText(coordinates_grids_x[column_i], draw.pixmap.height(),
                                  str(val_calibration_x[column_i]))
