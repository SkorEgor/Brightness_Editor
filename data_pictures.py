import numpy as np
import pandas as pd


# Линейная гистограмма по переданному изображению.
# Индексы - цвет; Значения - количество
def linear_histogram_from_picture(gray_image):
    # Создаем массив размера до максимального значения серого на картинке
    quantity = np.zeros(int(gray_image.max() + 1), dtype="int64")
    quantity = pd.Series(quantity)

    # Перебираем и считаем количество
    for i in gray_image:
        for j in i:
            quantity[int(j)] += 1

    return quantity


# Кумулятивная гистограмма по переданному изображению.
# Индексы - цвет; Значения - количество пикселей этого цвета + всех предыдущих
def cumulative_histogram_from_linear_histogram(linear_histogram):
    # Создаем массив размера до максимального значения серого на картинке
    cumulative_histograms = linear_histogram.copy()
    for i in range(1, linear_histogram.size):
        cumulative_histograms[i] += cumulative_histograms[i - 1]
    return cumulative_histograms


# Поиск коэффициентов для линейного растяжения
def search_coefficients_for_stretch(image):
    f_old_val_min = np.min(image)
    f_old_val_max = np.max(image)

    g_new_val_min = 0
    g_new_val_max = 255

    b = (g_new_val_max - g_new_val_min) / (f_old_val_max - f_old_val_min)
    a = g_new_val_min - b * f_old_val_min

    return a, b


# Нормировка гистограмм. Сумма = 1
def histogram_normalization(in_histogram):
    coefficient = 1 / in_histogram.sum()
    return in_histogram * coefficient


# КЛАСС ИЗОБРАЖЕНИЯ И ЕГО ХАРАКТЕРИСТИК
class DataPicture:
    # Конструктор
    def __init__(self):
        self.picture = None
        self.linear_histogram = None
        self.cumulative_histograms = None

    # Сохраняет переданное изображение и считает его гистограммы
    def saving_picture_and_calculating_histogram(self, input_picture):
        self.picture = input_picture
        self.update_histograms()

    # Перевычисляет гистограммы
    def update_histograms(self):
        self.linear_histogram = linear_histogram_from_picture(self.picture)
        self.cumulative_histograms = cumulative_histogram_from_linear_histogram(
            self.linear_histogram
        )

    # МЕТОДЫ
    # (1) Линейное растяжение гистограммы
    def stretch(self):
        # Находим конфиденты
        a, b = search_coefficients_for_stretch(self.picture)
        print(type(self.picture[0][0]))

        # Новая картинка считается по формуле:
        # new_image[i, j] = a + b * self.grey_picture[i, j]
        self.picture = a + b * self.picture

    # (2) Нормализация
    def normalization(self, percentage_saving, original_picture):
        if original_picture.picture is None:
            return

        original_picture.linear_histogram = linear_histogram_from_picture(original_picture.picture)
        original_picture.cumulative_histograms = cumulative_histogram_from_linear_histogram(
            original_picture.linear_histogram
        )

        if self.picture is None:
            self.picture = original_picture.picture.copy()

        self.linear_histogram = linear_histogram_from_picture(self.picture)


        # (1) Берем процент энергии исходной гистограммы
        # находим начало и конец интервала
        subinterval_of_nonzero_values = self.linear_histogram.loc[self.linear_histogram > 0]
        print(subinterval_of_nonzero_values)
        old_val_min = subinterval_of_nonzero_values.index[0]
        old_val_max = subinterval_of_nonzero_values.index[-1]
        old_len = subinterval_of_nonzero_values.size

        how_much_remove = old_len * percentage_saving / 100

        # Обнуление малоинформативных хвостов
        for i in range(int(how_much_remove // 2)):
            self.linear_histogram[old_val_min + i] = 0
            self.linear_histogram[old_val_max - i] = 0

        self.cumulative_histograms = cumulative_histogram_from_linear_histogram(self.linear_histogram)

        # Нормализуем гистограммы
        # #original_picture.linear_histogram = histogram_normalization(original_picture.linear_histogram)
        original_picture.cumulative_histograms = histogram_normalization(original_picture.cumulative_histograms)
        #
        # #self.linear_histogram = histogram_normalization(self.linear_histogram)
        self.cumulative_histograms = histogram_normalization(self.cumulative_histograms)

        self.update_cumulative_histograms_to_picture(original_picture)

    def update_cumulative_histograms_to_picture(self, original_picture):
        height, width = self.picture.shape

        print(type(self.picture[0,0]))
        # Уберем из рассмотрения нулевые элементы
        old_cumulative_histograms = original_picture.cumulative_histograms.loc[original_picture.cumulative_histograms > 0]
        new_cumulative_histograms = self.cumulative_histograms.loc[self.cumulative_histograms > 0]

        for i in range(height):
            for j in range(width):
                old_value_pixel = int(self.picture[i, j])
                count_in_linear_histogram = old_cumulative_histograms[old_value_pixel]
                new_value_pixel = abs(new_cumulative_histograms - count_in_linear_histogram).sort_values().index[0]
                # new_value_pixel = cumulative_histograms.loc[
                #     (cumulative_histograms - count_in_linear_histogram) < 0].index[-1]

                #print("old_value_pixel", old_value_pixel, "new_value_pixel", new_value_pixel, j,i)
                self.picture[i, j] = new_value_pixel

                # print(count_in_linear_histogram, cumulative_histograms.loc[
                #           (cumulative_histograms - count_in_linear_histogram) > 0]
                #       )


