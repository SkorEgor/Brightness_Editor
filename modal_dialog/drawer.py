from modal_dialog.graph import Graph


# ШАБЛОНЫ ОТРИСОВКИ ГРАФИКОВ
# Очистка и подпись графика (вызывается в начале)
def cleaning_and_chart_graph(graph: Graph):
    graph.toolbar.home()  # Возвращаем зум в домашнюю позицию
    graph.toolbar.update()  # Очищаем стек осей (от старых x, y lim)
    # Очищаем график
    graph.axis.clear()


# Отрисовка (вызывается в конце)
def draw_graph(graph: Graph):
    # Рисуем сетку
    graph.axis.grid()
    # Убеждаемся, что все помещается внутри холста
    graph.figure.tight_layout()
    # Показываем новую фигуру в интерфейсе
    graph.canvas.draw()


# Отрисовка при отсутствии данных
def no_data(graph: Graph):
    graph.axis.text(0.5, 0.5, "Нет данных",
                    fontsize=14,
                    horizontalalignment='center',
                    verticalalignment='center')
    # Отрисовка, без подписи данных графиков
    draw_graph(graph)


# Класс художник. Имя холст (graph), рисует на нем данные (data_and_processing)
class Drawer:
    # Отрисовка графика 2d
    @staticmethod
    def graph(
            graph: Graph,
            data
    ):
        # Очистка, подпись графика и осей (вызывается в начале)
        cleaning_and_chart_graph(
            # Объекты графика
            graph=graph
        )

        graph.axis.plot(data, color="#310DEC")

        # Отрисовка (вызывается в конце)
        draw_graph(graph)
