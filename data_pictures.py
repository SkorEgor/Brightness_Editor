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
        self.picture = None  # Изображение
        self.linear_histogram = None  # Линейная гистограмма
        self.cumulative_histogram = None  # Кумулятивная гистограмма с нормировкой

    # МЕТОДЫ ДЛЯ РАБОТЫ С ДАННЫМИ
    # (*) Сохраняет переданное изображение и считает его гистограммы
    def saving_picture_and_calculating_histogram(self, input_picture):
        self.picture = input_picture
        self.update_histograms()

    # (*) Вычисляет линейную гистограмму из исходного изображения
    def update_linear_histogram(self):
        self.linear_histogram = linear_histogram_from_picture(self.picture)

    # (*) Вычисляет НОРМИРОВАННУЮ кумулятивную гистограмму на основе линейной
    def update_cumulative_histogram(self):
        self.cumulative_histogram = histogram_normalization(  # Нормировка
            cumulative_histogram_from_linear_histogram(  # Построение на основе линейной гистограммы
                self.linear_histogram  # Исходные данные - собственная линейная гистограмма
            )
        )

    # (*) Вычисляет гистограммы
    def update_histograms(self):
        self.update_linear_histogram()  # Линейная из исходного изображения
        self.update_cumulative_histogram()  # Кумулятивная нормированная из линейной

    # МЕТОДЫ ДЛЯ РАБОТЫ ОБРАБОТКИ
    # (1) Линейное растяжение гистограммы
    def stretch(self):
        # Находим конфиденты
        print(type(self.picture[0][0]))
        a, b = search_coefficients_for_stretch(self.picture)

        # Новая картинка считается по формуле:
        # new_image[i, j] = a + b * self.grey_picture[i, j]
        self.picture = a + b * self.picture

    # (2) Нормализация
    @staticmethod
    def normalization(percentage_saving, original_picture):
        resulting_picture = DataPicture()

        # 1. Нет оригинального изображения - сброс
        if original_picture.picture is None:
            return

        # 2. Нет гистограмм оригинального изображения - рассчитать
        if (original_picture.linear_histogram is None) or (original_picture.cumulative_histogram is None):
            original_picture.update_histograms()

        # 3. Копируем линейную гистограмму исходного изображения
        resulting_picture.linear_histogram = original_picture.linear_histogram.copy()

        # 4. Обрезаем гистограмму результирующего изображения, по проценту сохраненных отсчетов
        # 4.1. Находим начало и конец интервала
        subinterval_of_nonzero_values = resulting_picture.linear_histogram.loc[
            resulting_picture.linear_histogram > 0]
        old_val_min = subinterval_of_nonzero_values.index[0]
        old_val_max = subinterval_of_nonzero_values.index[-1]
        old_len = subinterval_of_nonzero_values.size

        # 4.2. Результирующее количество отсчетов
        how_much_remove = old_len * percentage_saving / 100

        # 4.3. Обнуление малоинформативных хвостов
        for i in range(int(how_much_remove // 2)):
            resulting_picture.linear_histogram[old_val_min + i] = 0
            resulting_picture.linear_histogram[old_val_max - i] = 0

        # 5. Вычислили новую линейную гистограмму, по ней найдем кумулятивную гистограмму
        resulting_picture.update_cumulative_histogram()

        # 6. Исходным данным и гистограмме результирующего изображения.
        # Находим результирующее изображение
        resulting_picture.picture = DataPicture.cumulative_histograms_to_picture(
            resulting_picture.cumulative_histogram, original_picture)

        return resulting_picture

    # Метод создания изображения на основе
    # * Необходимой Кумулятивной гистограммы (cumulative_histogram_resulting) по
    # * Исходному изображению и Кумулятивная гистограмме
    @staticmethod
    def cumulative_histograms_to_picture(cumulative_histogram_resulting, original_picture):
        height, width = original_picture.picture.shape

        # 1. Создаем пустое результирующее изображение
        resulting_image = np.zeros((height, width), dtype="uint8")  # Результирующее изображение такого же размера

        # 2. Создаем массив перевода старых значений в новые
        old_to_new_value_array = np.zeros(256, dtype="uint8")
        for old_value_pixel in range(256):
            count_in_histogram = original_picture.cumulative_histogram[old_value_pixel]
            old_to_new_value_array[old_value_pixel] = (abs(cumulative_histogram_resulting - count_in_histogram)
                                                       ).sort_values().index[0]
        # Frame для проверки данных
        # print( pd.DataFrame({
        #     "orig":original_picture.cumulative_histogram,
        #     "res": cumulative_histogram_resulting,
        #     "old_to_new": old_to_new_value_array
        # }).to_string())

        # 3. Перевод
        for i in range(height):
            for j in range(width):
                resulting_image[i, j] = old_to_new_value_array[int(original_picture.picture[i, j])]

        # 4. Возвращаем результат
        return resulting_image

