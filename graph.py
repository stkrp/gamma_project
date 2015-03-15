__author__ = (
    'Egor Stahovskii <y.stakhovsky@gmail.com>: adaptation of algorithm, '
    'Stanislav Karpov <stkrp@yandex.ru>: coding & adaptation of algorithm, '
    'Fedor Zabrodin <iamfedor93@gmail.com>: adaptation of algorithm'
)

from copy import deepcopy


# TODO: Добавить проверку на мультиграф

class Graph(object):
    """ Граф """
    def __init__(self, *, adjacency_matrix=None, chain=None, type_=None, size=None):
        if adjacency_matrix is not None:
            self.ADJACENCY_MATRIX = deepcopy(adjacency_matrix)
        elif not any(param is None for param in (chain, type_, size)):
            self.ADJACENCY_MATRIX = Graph.create_empty_matrix(size)
            if len(chain) > 1:
                for i in range(len(chain[:-1])):
                    self.create_edge(chain[i], chain[i + 1])
                if type_ == 'cycle':
                    self.create_edge(chain[-1], chain[0])
            elif len(chain):
                self.create_edge(chain[0], chain[0])
            else:
                raise ValueError('Указана пустая цепь')
        else:
            raise ValueError('Неправильно заданы параметры инициализации графа')

    def __str__(self):
        result = ''
        for i, row in enumerate(self.ADJACENCY_MATRIX):
            result += '{:3}: {}\n'.format(i, row)

        return result

    def get_size(self):
        """ Получить размер матрицы смежности, т. е. количество вершин """
        return len(self.ADJACENCY_MATRIX)

    @staticmethod
    def create_empty_matrix(size):
        return [deepcopy([0] * size) for i in range(size)]

    def __add__(self, other):
        """ Объединение графов """
        if self.get_size() == other.get_size():
            result_matrix = deepcopy(self.ADJACENCY_MATRIX)
            size = len(result_matrix)
            for i in range(size):
                for j in range(size):
                    result_matrix[i][j] = int(result_matrix[i][j] or other.ADJACENCY_MATRIX[i][j])
            return Graph(adjacency_matrix=result_matrix)
        else:
            return None

    def __mul__(self, other):
        """ Пересечение графов """
        if self.get_size() == other.get_size():
            result_matrix = deepcopy(self.ADJACENCY_MATRIX)
            size = len(result_matrix)
            for i in range(size):
                for j in range(size):
                    result_matrix[i][j] = int(result_matrix[i][j] and other.ADJACENCY_MATRIX[i][j])
            return Graph(adjacency_matrix=result_matrix)
        else:
            return None

    def __sub__(self, other):
        """ Разность графов """
        if self.get_size() == other.get_size():
            result_matrix = deepcopy(self.ADJACENCY_MATRIX)
            size = len(result_matrix)
            for i in range(size):
                for j in range(size):
                    result_matrix[i][j] = int(result_matrix[i][j] and not other.ADJACENCY_MATRIX[i][j])
            return Graph(adjacency_matrix=result_matrix)
        else:
            return None

    def _set_edge_value(self, start_vertex, finish_vertex, value):
        """ Установить значение для ребра """
        self.ADJACENCY_MATRIX[start_vertex][finish_vertex] = value
        self.ADJACENCY_MATRIX[finish_vertex][start_vertex] = value

    def create_edge(self, start_vertex, finish_vertex):
        """ Создать ребро """
        self._set_edge_value(start_vertex, finish_vertex, 1)

    def delete_edge(self, start_vertex, finish_vertex):
        """ Удалить ребро """
        self._set_edge_value(start_vertex, finish_vertex, 0)

    def has_edge(self, start_vertex, finish_vertex):
        """ Проверить существования ребра """
        return bool(self.ADJACENCY_MATRIX[start_vertex][finish_vertex])

    def is_empty(self):
        """ Проверить граф на пустоту """
        return not bool(sum(sum(row) for row in self.ADJACENCY_MATRIX))

    def check_symmetry(self):
        """ Проверка квадратной матрицы на сиимметричность относительно главной диагонали """
        size = self.get_size()
        for i in range(size):
            for j in range(size):
                if i != j:
                    if self.ADJACENCY_MATRIX[i][j] != self.ADJACENCY_MATRIX[j][i]:
                        print('Матрица смежности не симметрична:', i, j)
                        return False
        return True

    def create_empty_adjacency_matrix_copy(self):
        """ Создать пустую копию матрицы смежности """
        return Graph.create_empty_matrix(self.get_size())

    def get_vertices(self):
        """ Получить вершины графа со степенью > 0 """
        size = self.get_size()
        return [i for i in range(size) if sum(self.ADJACENCY_MATRIX[i]) > 0]

    def get_adjacent_vertices(self, vertex):
        """ Получить список связных вершин """
        size = self.get_size()
        return [i for i in range(size) if self.ADJACENCY_MATRIX[vertex][i]]

    def find_simple_cycle(self, start_vertex=0, path=None):
        """ Поиск простого цикла """
        path = [] if path is None else deepcopy(path)
        adjacent_vertices = self.get_adjacent_vertices(start_vertex)
        path.append(start_vertex)
        for vertex in adjacent_vertices:
            if vertex in path[:-1]:
                if vertex == path[-2]:
                    continue
                else:
                    vertex_first_inclusion = path.index(vertex)
                    return path[vertex_first_inclusion:]
            else:
                next_step = self.find_simple_cycle(vertex, path)
                if isinstance(next_step, list):
                    return next_step

        return None

    def find_simple_chain(self, start_vertex, finish_vertex, path=None):
        """ Поиск простой цепи между двумя вершинами """
        path = [] if path is None else deepcopy(path)
        adjacent_vertices = self.get_adjacent_vertices(start_vertex)
        path.append(start_vertex)
        for vertex in adjacent_vertices:
            if vertex in path[:-1]:
                continue
            else:
                if vertex == finish_vertex:
                    path.append(vertex)
                    return path
                next_step = self.find_simple_chain(vertex, finish_vertex, path)
                if isinstance(next_step, list):
                    return next_step

        return None

    def check_coherence(self, start_vertex):
        """ Проверить граф на связность """
        path = [start_vertex]
        i = 0
        size = self.get_size()
        while len(path) != size:
            if i >= len(path):
                return False
            adjacent_vertices = self.get_adjacent_vertices(path[i])
            path.extend(list(set(adjacent_vertices) - set(path)))
            i += 1
        return True

    def is_simple_cycle(self):
        """ Является ли граф циклом """
        graph_vertices = self.get_vertices()
        if graph_vertices:
            start_vertex = graph_vertices[0]
            cycle = self.find_simple_cycle(start_vertex)
            cycle_graph = Graph(chain=cycle, type_='cycle', size=self.get_size())
            if (self - cycle_graph).is_empty():
                return True
            else:
                return False
        else:
            print('Граф пуст')
            return False


class GammaGraph(Graph):
    """ 'Плоский' граф (граф, полученный в процессе укладки Graph) """
    def __init__(self, original_graph):
        self.original_graph = original_graph
        Graph.__init__(
            self,
            chain=self.original_graph.find_simple_cycle(),
            type_='cycle',
            size=self.original_graph.get_size()
        )
        self._segments_graph = self.original_graph - self

    def get_contact_vertices(self):
        """ Получить список контактных вершин """
        size = self.get_size()
        return [i for i in range(size) if self.ADJACENCY_MATRIX[i]]

    def _get_macrosegment(self, start_vertex, segment=None):
        """ Получить сегмент с неконтактными вершинами """
        if segment is None:
            segment = Segment(adjacency_matrix=self.create_empty_adjacency_matrix_copy(), gamma_graph=self)
        adjacent_vertices = self._segments_graph.get_adjacent_vertices(start_vertex)
        contact_vertices = self.get_vertices()
        for adjacent_vertex in adjacent_vertices:
            if self._segments_graph.has_edge(start_vertex, adjacent_vertex):
                segment.create_edge(start_vertex, adjacent_vertex)
                self._segments_graph.delete_edge(start_vertex, adjacent_vertex)
                if adjacent_vertex not in contact_vertices:
                    segment = self._get_macrosegment(adjacent_vertex, segment)
        return segment

    def _get_minisegment(self, start_vertex, finish_vertex):
        """ Получить сегмент, состоящий из одного ребра """
        contact_vertices = self.get_vertices()
        if (
            start_vertex in contact_vertices and
            finish_vertex in contact_vertices and
            self._segments_graph.has_edge(start_vertex, finish_vertex)
        ):
            segment = Segment(adjacency_matrix=self.create_empty_matrix(self.get_size()), gamma_graph=self)
            segment.create_edge(start_vertex, finish_vertex)
            self._segments_graph.delete_edge(start_vertex, finish_vertex)
            return segment
        else:
            return Graph(adjacency_matrix=self.create_empty_adjacency_matrix_copy())

    def _get_segments(self):
        segments = []
        self._segments_graph = self.original_graph - self
        contact_vertices = self.get_vertices()
        while True:
            for contact_vertex in contact_vertices:
                adjacent_vertices = self._segments_graph.get_adjacent_vertices(contact_vertex)
                for adjacent_vertex in adjacent_vertices:
                    if adjacent_vertices in contact_vertices:
                        minisegment = self._get_minisegment(contact_vertex, adjacent_vertex)
                        if not minisegment.is_empty():
                            segments.append(minisegment)
                macrosegment = self._get_macrosegment(contact_vertex)
                if not macrosegment.is_empty():
                    segments.append(macrosegment)
            if self._segments_graph.is_empty():
                break

        return segments


class Face(Graph):
    def split(self, chain):
        if len(chain) > 1:
            start_vertex = chain[0]
            finish_vertex = chain[-1]
            """ Разбить грань на 2 грани по цепи между двумя вершинами """
            if not (start_vertex == finish_vertex or self.has_edge(start_vertex, finish_vertex)):
                if self.is_simple_cycle():
                    first_face = self.__class__(
                        chain=self.find_simple_chain(start_vertex, finish_vertex),
                        type_='chain',
                        size=self.get_size()
                    )
                    second_face = self - first_face

                    chain_graph = Graph(chain=chain, type_='chain', size=self.get_size())
                    first_face += chain_graph
                    second_face += chain_graph
                    return first_face, second_face
                else:
                    print('Граф не является простым циклом')
                    return None
            else:
                print('Вершины начала и конца совпадают или смежны')
                return None
        else:
            print('Цепь состоит менее чем из 2-х вершин')
            return None


class Segment(Graph):
    """ Сегмент """
    def __init__(self, *, adjacency_matrix=None, chain=None, type_=None, size=None, gamma_graph=None):
        # Не забыть про матрицу смежности для сегмента
        self.gamma_graph = gamma_graph
        Graph.__init__(self, adjacency_matrix=adjacency_matrix, chain=chain, type_=type_, size=size)


if __name__ == '__main__':
    import sys
    sys.stdout = open('output.txt', 'wt', encoding='utf-8')

    matrix = [
        [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    ]

    matrix2 = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    ]

    g1 = Graph(adjacency_matrix=matrix)
    # print('split', g1.split(1, 4), sep='\n')
    gg1 = GammaGraph(g1)
    face1 = Face(adjacency_matrix=gg1.ADJACENCY_MATRIX)

    # print(gg1.is_simple_cycle())

    # print('Start cycle:', gg1.get_vertices())
    # print(*gg1._get_segments(), sep='\n')
    print(gg1)
    print('split', *face1.split([1, 6, 8, 4]), sep='\n')