__author__ = (
    'Zabrodin Fedor <iamfedor93@gmail.com>: adaptation of algorithm, '
    'Karpov Stanislav <stkrp@yandex.ru>: coding & adaptation of algorithm, '
    'Stahovskii Egor <y.stakhovsky@gmail.com>: adaptation of algorithm'
)

from copy import deepcopy


# TODO: Добавить проверку на мультиграф
# TODO: Что делать, если есть точки сочленения?

class Graph(object):
    """ Граф """
    @staticmethod
    def create_empty_matrix(len_):
        return [deepcopy([0] * len_) for i in range(len_)]

    def __init__(self, *, adjacency_matrix=None, chain=None, type_=None, len_=None):
        if adjacency_matrix is not None:
            self.ADJACENCY_MATRIX = deepcopy(adjacency_matrix)
        elif not any(param is None for param in (chain, type_, len_)):
            self.ADJACENCY_MATRIX = Graph.create_empty_matrix(len_)
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

    def __eq__(self, other):
        """ Проверить равенство графов """
        return self.ADJACENCY_MATRIX == other.ADJACENCY_MATRIX

    def __len__(self):
        """ Получить размер матрицы смежности, т. е. количество вершин """
        return len(self.ADJACENCY_MATRIX)

    def __add__(self, other):
        """ Объединение графов """
        if len(self) == len(other):
            result_matrix = deepcopy(self.ADJACENCY_MATRIX)
            len_ = len(self)
            for i in range(len_):
                for j in range(len_):
                    result_matrix[i][j] = int(result_matrix[i][j] or other.ADJACENCY_MATRIX[i][j])
            return self.__class__(adjacency_matrix=result_matrix)
        else:
            print('Графы разных размеров')
            return self.__class__(adjacency_matrix=self.create_empty_adjacency_matrix_copy())

    def __mul__(self, other):
        """ Пересечение графов """
        if len(self) == len(other):
            result_matrix = deepcopy(self.ADJACENCY_MATRIX)
            len_ = len(self)
            for i in range(len_):
                for j in range(len_):
                    result_matrix[i][j] = int(result_matrix[i][j] and other.ADJACENCY_MATRIX[i][j])
            return self.__class__(adjacency_matrix=result_matrix)
        else:
            print('Графы разных размеров')
            return self.__class__(adjacency_matrix=self.create_empty_adjacency_matrix_copy())

    def __sub__(self, other):
        """ Разность графов """
        if len(self) == len(other):
            result_matrix = deepcopy(self.ADJACENCY_MATRIX)
            len_ = len(self)
            for i in range(len_):
                for j in range(len_):
                    result_matrix[i][j] = int(result_matrix[i][j] and not other.ADJACENCY_MATRIX[i][j])
            return self.__class__(adjacency_matrix=result_matrix)
        else:
            print('Графы разных размеров')
            return self.__class__(adjacency_matrix=self.create_empty_adjacency_matrix_copy())

    def __contains__(self, item):
        """ Проверка графа item на подграф self """
        if len(self) == len(item):
            len_ = len(self)
            for i in range(len_):
                for j in range(len_):
                    if item.ADJACENCY_MATRIX[i][j] > self.ADJACENCY_MATRIX[i][j]:
                        return False
            return True
        else:
            print('Графы разных размеров')
            return False

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
        len_ = len(self)
        for i in range(len_):
            for j in range(len_):
                if i != j:
                    if self.ADJACENCY_MATRIX[i][j] != self.ADJACENCY_MATRIX[j][i]:
                        print('Матрица смежности не симметрична:', i, j)
                        return False
        return True

    def create_empty_adjacency_matrix_copy(self):
        """ Создать пустую копию матрицы смежности """
        return Graph.create_empty_matrix(len(self))

    def get_vertices(self):
        """ Получить вершины графа со степенью > 0 """
        len_ = len(self)
        return {i for i in range(len_) if sum(self.ADJACENCY_MATRIX[i]) > 0}

    def get_adjacent_vertices(self, vertex):
        """ Получить набор связных вершин """
        len_ = len(self)
        if vertex < len_:
            return {i for i in range(len_) if self.ADJACENCY_MATRIX[vertex][i]}
        else:
            return set()

    def get_simple_cycle(self, start_vertex, path=None, *, save_start_vertex=False):
        """ Поиск простого цикла """
        path = [] if path is None else deepcopy(path)
        adjacent_vertices = self.get_adjacent_vertices(start_vertex)
        path.append(start_vertex)
        for vertex in adjacent_vertices:
            if vertex in path[:-1]:
                if vertex == path[-2]:
                    continue
                else:
                    if not save_start_vertex:
                        vertex_first_inclusion = path.index(vertex)
                        return path[vertex_first_inclusion:]
                    else:
                        if vertex == path[0]:
                            return path
                        continue
            else:
                next_step = self.get_simple_cycle(vertex, path, save_start_vertex=save_start_vertex)
                if isinstance(next_step, list) and next_step:
                    return next_step

        return []

    def get_simple_chain(self, start_vertex, finish_vertex, path=None):
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
                next_step = self.get_simple_chain(vertex, finish_vertex, path)
                if isinstance(next_step, list) and next_step:
                    return next_step

        return []

    def check_coherence(self, start_vertex):
        """ Проверить граф на связность """
        path = [start_vertex]
        i = 0
        len_ = len(self)
        while len(path) != len_:
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
            start_vertex = graph_vertices.pop()
            cycle = self.get_simple_cycle(start_vertex)
            if cycle:
                cycle_graph = Graph(chain=cycle, type_='cycle', len_=len(self))
                if (self - cycle_graph).is_empty():
                    return True
                else:
                    return False
            else:
                return False
        else:
            print('Граф пуст')
            return False


class GammaGraph(Graph):
    """ 'Плоский' граф (граф, полученный в процессе укладки Graph) """
    def __init__(self, *, adjacency_matrix=None, chain=None, type_=None, len_=None, original_graph=None):
        if original_graph is None:
            Graph.__init__(self, adjacency_matrix=adjacency_matrix, chain=chain, type_=type_, len_=len_)
            self._original_graph = self.create_empty_adjacency_matrix_copy()
            self.faces = [Face(adjacency_matrix=self.ADJACENCY_MATRIX) for i in range(2)]
        # TODO: Добавить все проверки original_graph вместо True
        elif True:
            self._original_graph = deepcopy(original_graph)
            Graph.__init__(
                self,
                chain=self._original_graph.get_simple_cycle(0),
                type_='cycle',
                len_=len(self._original_graph)
            )
            self.faces = [Face(adjacency_matrix=self.ADJACENCY_MATRIX) for i in range(2)]
            while True:
                self._segments = self._get_segments()
                if self._segments:
                    # Минимальный сегмент - сегмент, количество вмещающих граней которого минимального
                    # Поиск минимального сегмента (при условии, что для каждого сегмента есть вмещающая грань)
                    min_segment = self._segments[0]
                    if not min_segment.get_inclusive_faces():
                        raise ValueError('Граф непланарный')
                    for segment in self._segments[1:]:
                        segment_inclusive_faces = segment.get_inclusive_faces()
                        if not segment_inclusive_faces:
                            raise ValueError('Граф непланарный')
                        elif len(segment_inclusive_faces) < len(min_segment.get_inclusive_faces()):
                            min_segment = segment
                    # Выбор грани для минмального сегмента
                    for i, face in enumerate(self.faces):
                        # Получение 2-х новых граней из выбранной, полученных при помощи цепи минимального сегмента
                        if min_segment.included_in_face(face):
                            min_segment_contact_vertices = min_segment.get_contact_vertices()
                            for start_contact_vertex in min_segment_contact_vertices:
                                min_segment_contact_vertices_without_start_vertex = (
                                    min_segment_contact_vertices - {start_contact_vertex}
                                )
                                if not min_segment_contact_vertices_without_start_vertex:
                                    min_segment_chain = min_segment.get_simple_cycle(
                                        start_contact_vertex, None, save_start_vertex=True
                                    )
                                    if min_segment_chain:
                                        new_face = Face(chain=min_segment_chain, type_='cycle', len_=len(self))
                                        self.faces.extend(deepcopy(new_face) for i in range(2))
                                        gamma_graph_update = Graph(adjacency_matrix=new_face.ADJACENCY_MATRIX)
                                        break
                                else:
                                    for finish_contact_vertex in min_segment_contact_vertices_without_start_vertex:
                                        min_segment_chain = (
                                            min_segment.get_simple_chain_without_contact_vertices(
                                                start_contact_vertex, finish_contact_vertex
                                            )
                                        )
                                        if not min_segment_chain:
                                            continue
                                        new_faces = face.split(min_segment_chain)
                                        if new_faces:
                                            del self.faces[i]
                                            self.faces.extend(new_faces)
                                            gamma_graph_update = Graph(
                                                chain=min_segment_chain, type_='chain', len_=len(self)
                                            )
                                            break
                                    else:
                                        continue
                                    break
                            else:
                                raise RuntimeError(
                                    'Не найдены вершины для вписывания цепи сегмента в грань'
                                )

                            self.ADJACENCY_MATRIX = (
                                (Graph(adjacency_matrix=self.ADJACENCY_MATRIX) + gamma_graph_update).ADJACENCY_MATRIX
                            )
                            break
                    else:
                        raise RuntimeError('Для сегмента не найдена грань')
                else:
                    break
            uniq_faces = []
            for face in self.faces:
                if face not in uniq_faces:
                    uniq_faces.append(face)
            self.faces = uniq_faces
        else:
            raise ValueError('Входной граф не удовлетворяет условиям алгоритма')

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
        segment = Segment(adjacency_matrix=self.create_empty_adjacency_matrix_copy(), gamma_graph=self)
        if (
            start_vertex in contact_vertices and
            finish_vertex in contact_vertices and
            self._segments_graph.has_edge(start_vertex, finish_vertex)
        ):
            segment.create_edge(start_vertex, finish_vertex)
            self._segments_graph.delete_edge(start_vertex, finish_vertex)
        return segment

    def _get_segments(self):
        segments = []
        self._segments_graph = self._original_graph - self
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

    def _get_vertex_sequence_for_paint(self, vertex, sequence=None, prev_adjacent_vertex=None):
        # TODO: Не готово
        """ Получить последовательность вершин """
        if sequence is None:
            sequence = []
        if prev_adjacent_vertex is None:
            prev_adjacent_vertex = set()
        adjacent_vertex = self.get_adjacent_vertices(vertex)
        next_vertices = adjacent_vertex - set(sequence) - prev_adjacent_vertex
        if next_vertices:
            next_vertex = next_vertices.pop()
            sequence.append(next_vertex)
            if self.get_vertices() == set(sequence):
                return sequence
            else:
                sequence = self._get_vertex_sequence_for_paint(next_vertex, sequence, next_vertices)

        # sequence.append()

    def get_faces_as_chains(self):
        """ Получить грани в виде цепей """
        return (face.get_simple_cycle(face.get_vertices().pop()) for face in self.faces if not face.is_empty())


class Face(Graph):
    """ Грань """
    def __init__(self, *, adjacency_matrix=None, chain=None, type_=None, len_=None, gamma_graph=None):
        self._gamma_graph = gamma_graph
        Graph.__init__(self, adjacency_matrix=adjacency_matrix, chain=chain, type_=type_, len_=len_)

    def split(self, chain):
        if len(chain) > 1:
            start_vertex = chain[0]
            finish_vertex = chain[-1]
            """ Разбить грань на 2 грани по цепи между двумя вершинами """
            if not (start_vertex == finish_vertex):
                if self.is_simple_cycle():
                    first_face = self.__class__(
                        chain=self.get_simple_chain(start_vertex, finish_vertex),
                        type_='chain',
                        len_=len(self)
                    )
                    second_face = self - first_face

                    chain_graph = Graph(chain=chain, type_='chain', len_=len(self))
                    first_face += chain_graph
                    second_face += chain_graph
                    return first_face, second_face
                else:
                    print('Граф не является простым циклом')
                    return ()
            else:
                print('Вершины начала и конца совпадают')
                return ()
        else:
            print('Цепь состоит менее чем из 2-х вершин')
            return ()


class Segment(Graph):
    """ Сегмент """
    def __init__(self, *, adjacency_matrix=None, chain=None, type_=None, len_=None, gamma_graph=None):
        self._gamma_graph = gamma_graph
        Graph.__init__(self, adjacency_matrix=adjacency_matrix, chain=chain, type_=type_, len_=len_)

    def get_contact_vertices(self):
        """ Получить набор контактных вершин """
        return set(self.get_vertices()) & set(self._gamma_graph.get_vertices())

    def included_in_face(self, face):
        """ Проверить, включен ли сегмент в грань """
        return self.get_contact_vertices() <= face.get_vertices()

    def get_inclusive_faces(self):
        """ Получить набор вмещающих граней """
        result = []
        for face in self._gamma_graph.faces:
            if self.included_in_face(face):
                result.append(face)
        return result

    def get_simple_chain_without_contact_vertices(self, start_vertex, finish_vertex, path=None):
        """ Получить простую цепь без контактных вершин """
        path = [] if path is None else deepcopy(path)
        adjacent_vertices = self.get_adjacent_vertices(start_vertex)
        contact_vertices = self.get_contact_vertices()
        path.append(start_vertex)
        vertices = adjacent_vertices - (contact_vertices - {finish_vertex})
        for vertex in vertices:
            if vertex in path[:-1]:
                continue
            else:
                if vertex == finish_vertex:
                    path.append(vertex)
                    return path
                next_step = self.get_simple_chain_without_contact_vertices(vertex, finish_vertex, path)
                if isinstance(next_step, list) and next_step:
                    return next_step

        return []

if __name__ == '__main__':
    import sys
    # sys.stdout = open('output.txt', 'wt', encoding='utf-8')

    matrix1 = [
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
        [0, 1, 1, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 0, 0],
    ]

    matrix3 = [
        [0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ]

    matrix4 = [
        [0, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ]

    # 2
    matrix5 = [
        [0, 1, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0],
    ]
    # 3
    matrix6 = [
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ]
    # 4
    matrix7 = [
        [0, 1, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 0],
    ]
    # 5
    matrix8 = [
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ]

    for k, matrix in enumerate((matrix1, matrix2, matrix3, matrix4, matrix5, matrix6, matrix7, matrix8)):
        try:
            graph = Graph(adjacency_matrix=matrix)
            ggraph = GammaGraph(original_graph=graph)
            print('Graph #{} faces:'.format(k), *ggraph.get_faces_as_chains(), sep='\n')
        except Exception as err:
            print('Graph #{} (error: {})'.format(k, err))

    # help(Graph)
    # help(GammaGraph)
    # help(Segment)
    # help(Face)