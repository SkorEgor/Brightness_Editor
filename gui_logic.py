import random

from PyQt5.QtWidgets import QFileDialog

from gui import Ui_Dialog

from graph import Graph
from drawer import Drawer as drawer
from data_pictures import DataPicture

from PyQt5 import QtCore
import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')


# Перевод цветной картинки в серую
def black_white_image(color_picture):
    height, width, _ = color_picture.shape

    gray_image = np.zeros((height, width), dtype="uint8")
    for i in range(height):
        for j in range(width):
            pixel = color_picture[i, j]
            gray_image[i, j] = np.uint8(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])

    maximum_intensity = np.max(gray_image)
    multiplier = 255 / maximum_intensity
    gray_image = gray_image * multiplier

    return gray_image

def search_coefficients_for_stretch(gray_image):
    f_old_val_min = np.min(gray_image)
    f_old_val_max = np.max(gray_image)

    g_new_val_min = 0
    g_new_val_max = 255

    b = (g_new_val_max - g_new_val_min) / (f_old_val_max - f_old_val_min)
    a = g_new_val_min - b * f_old_val_min

    return a, b

# Получение кумулятивной гистограммы
def get_cumulative_histograms(linear_histogram):
    cumulative_histograms = linear_histogram
    for i in range(1,linear_histogram.size):
        cumulative_histograms[i] += cumulative_histograms[i-1]
    return cumulative_histograms

# Гистограмма для переданного изображения. Индексы - цвет; Значения - количество
def histogram_from_picture(gray_image):
    # Создаем массив размера до максимального значения серого на картинке
    quantity = pd.Series([0] * int(gray_image.max()+1))

    # Перебираем и считаем количество
    for i in gray_image:
        for j in i:
            quantity[int(j)] += 1

    return quantity


# КЛАСС АЛГОРИТМА ПРИЛОЖЕНИЯ
class GuiProgram(Ui_Dialog):
    # Конструктор
    def __init__(self, dialog):
        # Создаем окно
        Ui_Dialog.__init__(self)
        dialog.setWindowFlags(  # Передаем флаги создания окна
            QtCore.Qt.WindowCloseButtonHint |  # Кнопка закрытия
            QtCore.Qt.WindowMaximizeButtonHint |  # Кнопка развернуть
            QtCore.Qt.WindowMinimizeButtonHint  # Кнопка свернуть
        )
        # Устанавливаем пользовательский интерфейс
        self.setupUi(dialog)

        # ПОЛЯ КЛАССА
        # Кумулятивная гистограмма исходного изображения
        self.graph_cumulative_histogram_original_picture = Graph(
            layout=self.layout_plot_1,
            widget=self.widget_plot_1,
            name_graphics="Рис. 1. Кумулятивная гистограмма",
            init_toolbar=False
        )
        # Линейная гистограмма исходного изображения
        self.graph_linear_histogram_original_picture = Graph(
            layout=self.layout_plot_2,
            widget=self.widget_plot_2,
            name_graphics="Рис. 2. Линейная гистограмма",
            init_toolbar=False
        )
        # Оригинальное изображение
        self.graph_original_picture = Graph(
            layout=self.layout_plot_3,
            widget=self.widget_plot_3,
            name_graphics="Рис. 3. Оригинальное изображение",
            init_toolbar=True
        )

        # Кумулятивная гистограмма результирующего изображения
        self.graph_cumulative_histogram_resulting_picture = Graph(
            layout=self.layout_plot_4,
            widget=self.widget_plot_4,
            name_graphics="Рис. 4. Кумулятивная гистограмма",
            init_toolbar=False
        )
        # Линейная гистограмма результирующего изображения
        self.graph_linear_histogram_resulting_picture = Graph(
            layout=self.layout_plot_5,
            widget=self.widget_plot_5,
            name_graphics="Рис. 5. Линейная гистограмма",
            init_toolbar=False
        )
        # Результирующее изображение
        self.graph_resulting_picture = Graph(
            layout=self.layout_plot_6,
            widget=self.widget_plot_6,
            name_graphics="Рис. 6. Результирующее изображение",
            init_toolbar=True
        )

        # Картинки обработки
        self.color_picture = None
        self.original_picture = DataPicture()
        self.resulting_picture = DataPicture()

        # ДЕЙСТВИЯ ПРИ ВКЛЮЧЕНИИ
        # Смена режима отображения картинки
        self.radioButton_color_picture.clicked.connect(self.change_picture_to_colored)
        self.radioButton_gray_picture.clicked.connect(self.change_picture_to_gray)

        # Алгоритм обратки
        # Загрузка картинки
        self.pushButton_loading_pictures.clicked.connect(self.load_image)
        self.pushButton_stretch.clicked.connect(self.stretch)
        self.pushButton_normalization.clicked.connect(self.normalization)

    # ОБРАБОТКА ИНТЕРФЕЙСА
    # Смена режима отображения картинки
    # Выбрано отображение цветной картинки
    def change_picture_to_colored(self, state):
        if state and self.color_picture is not None:
            drawer.image_color(self.graph_original_picture, self.color_picture)

    # Выбрано отображение серой картинки
    def change_picture_to_gray(self, state):
        if state and self.original_picture.picture is not None:
            drawer.image_gray(self.graph_original_picture, self.original_picture.picture)

    # Отобразить картинку
    def display_picture(self):
        # Картинки нет - не отображаем
        if self.color_picture is None:
            return

        # Проверяем вид отображаемой картинки
        if self.radioButton_color_picture.isChecked():
            drawer.image_color(self.graph_original_picture, self.color_picture)
        else:
            drawer.image_gray(self.graph_original_picture, self.original_picture.picture)

    # АЛГОРИТМ РАБОТЫ ПРОГРАММЫ
    # (0) Загрузить картинку
    def load_image(self):
        # Вызов окна выбора файла
        # filename, filetype = QFileDialog.getOpenFileName(None,
        #                                                  "Выбрать файл изображения",
        #                                                  ".",
        #                                                  "All Files(*)")
        filename = "image_256_128.png"

        # Загружаем картинку
        self.color_picture = cv2.imread(filename, cv2.IMREAD_COLOR)
        # Конвертируем в серый и заносим в класс исходного изображения, рассчитывая гистограммы
        self.original_picture.saving_picture_and_calculating_histogram(black_white_image(self.color_picture))
        # Отображаем
        self.display_data_picture_in_graph_original(self.original_picture)

    # Линейное растяжение гистограммы
    def stretch(self):
        self.resulting_picture.picture = self.original_picture.picture
        self.resulting_picture.stretch()
        self.resulting_picture.update_histograms()
        self.display_data_picture_in_graph_resulting(self.resulting_picture)

    def normalization(self):
        self.resulting_picture.picture = self.original_picture.picture.copy()

        # 1) Нормировка гистограммы
        self.resulting_picture.normalization(float(self.lineEdit_percent_normalization.text()),
                                             self.original_picture)

        #self.resulting_picture.update_histograms()
        self.display_data_picture_in_graph_resulting(self.resulting_picture)
        self.display_data_picture_in_graph_original(self.original_picture)


        # new_histogram = self.histogram_original_picture
        #
        # # Находим начало и конец интервала
        # subinterval_of_nonzero_values = self.histogram_original_picture.loc[self.histogram_original_picture > 0]
        # old_val_min = subinterval_of_nonzero_values.index[0]
        # old_val_max = subinterval_of_nonzero_values.index[-1]
        # old_len = subinterval_of_nonzero_values.size
        #
        # how_much_remove = old_len * float(self.lineEdit_percent_normalization.text()) / 100
        #
        # # Обнуление малоинформативных хвостов
        # for i in range(int(how_much_remove // 2)):
        #     new_histogram[old_val_min + i] = 0
        #     new_histogram[old_val_max - i] = 0
        #
        # drawer.histogram(self.graph_histogram_received_picture, new_histogram)

    # Вывод гистограмм и картинки, в оригинальный столбец.
    def display_data_picture_in_graph_original(self, picture: DataPicture):
        drawer.histogram(self.graph_cumulative_histogram_original_picture, picture.cumulative_histograms)
        drawer.histogram(self.graph_linear_histogram_original_picture, picture.linear_histogram)
        self.display_picture()

    # Вывод гистограмм и картинки, в результирующий столбец.
    def display_data_picture_in_graph_resulting(self, picture: DataPicture):
        drawer.histogram(self.graph_cumulative_histogram_resulting_picture, picture.cumulative_histograms)
        drawer.histogram(self.graph_linear_histogram_resulting_picture, picture.linear_histogram)
        drawer.image_gray(self.graph_resulting_picture, picture.picture)
