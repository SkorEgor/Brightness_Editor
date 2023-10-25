from gui import Ui_Dialog

from graph import Graph
from drawer import Drawer as drawer
from data_pictures import DataPicture

from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog
import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')


# Перевод цветной картинки в серую
def black_white_image(color_picture):
    # Получаем размер
    height, width, _ = color_picture.shape

    # Переводим в серый
    gray_image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            pixel = color_picture[i, j]
            gray_image[i, j] = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]

    # Нормируем максимум к 255
    maximum_intensity = np.max(gray_image)
    multiplier = 255 / maximum_intensity
    gray_image = gray_image * multiplier

    # Округляем
    gray_image = np.round(gray_image, decimals=0)
    # Привет к uint8 - беззнаковый int 8 байт - 255 max
    gray_image = gray_image.astype("uint8")

    return gray_image


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
        self.pushButton_normalization_percent_counts.clicked.connect(self.normalization_by_samples)
        self.pushButton_normalization_percent_counts_2.clicked.connect(self.normalization_by_intensity)

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
        # Нет изображения - сброс
        if self.original_picture.picture is None:
            return

        self.resulting_picture.picture = self.original_picture.picture
        self.resulting_picture.stretch()    # Считаем результирующее изображение
        self.resulting_picture.update_histograms()    # Для него обновляем гистограммы
        self.display_data_picture_in_graph_resulting(self.resulting_picture)    # Отображаем результат

    # Нормализация по отсчетам - обрезаем линейную гистограмму и восстанавливаем изображение
    def normalization_by_samples(self):
        # Нет изображения - сброс
        if self.original_picture.picture is None:
            return

        # 0) Нормализация по переданному проценту
        self.resulting_picture = DataPicture.normalization_by_samples(
            float(self.lineEdit_normalization_percent_counts.text()),
            self.original_picture)

        # 2) Отображаем результат
        self.display_data_picture_in_graph_resulting(self.resulting_picture)

    # Нормализация по интенсивности - обрезаем линейную гистограмму и восстанавливаем изображение
    def normalization_by_intensity(self):
        # Нет изображения - сброс
        if self.original_picture.picture is None:
            return

        # 0) Нормализация по переданному проценту
        self.resulting_picture = DataPicture.normalization_by_intensity(
            float(self.lineEdit_normalization_percent_counts_2.text()),
            self.original_picture)

        # 2) Отображаем результат
        self.display_data_picture_in_graph_resulting(self.resulting_picture)


    # Вывод гистограмм и картинки, в оригинальный столбец.
    def display_data_picture_in_graph_original(self, picture: DataPicture):
        drawer.histogram(self.graph_cumulative_histogram_original_picture, picture.cumulative_histogram)
        drawer.histogram(self.graph_linear_histogram_original_picture, picture.linear_histogram)
        self.display_picture()

    # Вывод гистограмм и картинки, в результирующий столбец.
    def display_data_picture_in_graph_resulting(self, picture: DataPicture):
        drawer.histogram(self.graph_cumulative_histogram_resulting_picture, picture.cumulative_histogram)
        drawer.histogram(self.graph_linear_histogram_resulting_picture, picture.linear_histogram)
        drawer.image_gray(self.graph_resulting_picture, picture.picture)
