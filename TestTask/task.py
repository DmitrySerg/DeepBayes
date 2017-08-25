# -*- coding: utf-8 -*-

import numpy as np
from scipy import special
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.datasets import make_classification

# Используйте scipy.special для вычисления численно неустойчивых функций
# https://docs.scipy.org/doc/scipy/reference/special.html#module-scipy.special

def lossf(w, X, y, l1, l2):
    """
    Вычисление функции потерь.

    :param w: numpy.array размера  (M,) dtype = np.float
    :param X: numpy.array размера  (N, M), dtype = np.float
    :param y: numpy.array размера  (N,), dtype = np.int
    :param l1: float, l1 коэффициент регуляризатора 
    :param l2: float, l2 коэффициент регуляризатора 
    :return: float, value of loss function
    """
    lossf = np.sum(np.log(1 + np.exp((-y*np.dot(X, w))))) + l1*np.sum(np.abs(w)) + l2*np.sum(np.power(w, 2))
    return lossf

def gradf(w, X, y, l1, l2):
    """
    Вычисление градиента функции потерь.

    :param w: numpy.array размера  (M,), dtype = np.float
    :param X: numpy.array размера  (N, M), dtype = np.float
    :param y: numpy.array размера  (N,), dtype = np.int
    :param l1: float, l1 коэффициент регуляризатора 
    :param l2: float, l2 коэффициент регуляризатора 
    :return: numpy.array размера  (M,), dtype = np.float, gradient vector d lossf / dw
    """
    gradw = np.dot((1/(1+np.exp((-y*np.dot(X, w))))),(np.exp((-y*np.dot(X, w)))*(-y*X.T)).T)+l1*np.sign(w)+l2*2*w
    return gradw

class LR(ClassifierMixin, BaseEstimator):
    def __init__(self, lr=1, l1=1e-4, l2=1e-4, num_iter=1000, verbose=0):
        self.l1 = l1
        self.l2 = l2
        self.w = None
        self.lr = lr
        self.verbose = verbose
        self.num_iter = num_iter

    def fit(self, X, y):
        """
        Обучение логистической регрессии.
        Настраивает self.w коэффициенты модели.

        Если self.verbose == True, то выводите значение 
        функции потерь на итерациях метода оптимизации. 

        :param X: numpy.array размера  (N, M), dtype = np.float
        :param y: numpy.array размера  (N,), dtype = np.int
        :return: self
        """
        n, d = X.shape
        self.w = np.random.uniform(size=100).reshape((100,))
        for i in range(self.num_iter):
            grad = gradf(self.w, X, y, self.l1, self.l2)
            
            while lossf(self.w-grad*self.lr, X, y, self.l1, self.l2) >\
                  lossf(self.w, X, y, self.l1, self.l2) - 0.1*self.lr*np.linalg.norm(grad, ord=2)**2:
                    
                self.lr = 0.95*self.lr
            self.w -= grad * self.lr 
 
            if self.verbose:
                print lossf(self.w, X, y, self.l1, self.l2)

        return self

    def predict_proba(self, X):
        """
        Предсказание вероятности принадлежности объекта к классу 1.
        Возвращает np.array размера (N,) чисел в отрезке от 0 до 1.

        :param X: numpy.array размера  (N, M), dtype = np.float
        :return: numpy.array размера  (N,), dtype = np.int
        """
        # Вычислите вероятности принадлежности каждого 
        # объекта из X к положительному классу, используйте
        # эту функцию для реализации LR.predict
        odds = np.exp(np.dot(X, self.w))
        probs = odds/(1+odds)
        return probs

    def predict(self, X):
        """
        Предсказание класса для объекта.
        Возвращает np.array размера (N,) элементов 1 или -1.

        :param X: numpy.array размера  (N, M), dtype = np.float
        :return:  numpy.array размера  (N,), dtype = np.int
        """
        # Вычислите предсказания для каждого объекта из X
        predicts = np.array([1 if i else -1 for i in self.predict_proba(X)>0.5])
        return predicts 

def test_work():
    print "Start test"
    X, y = make_classification(n_features=100, n_samples=1000)
    y = 2 * (y - 0.5)

    try:
        clf = LR(lr=1, l1=1e-4, l2=1e-4, num_iter=1000, verbose=0)
    except Exception:
        assert False, "Создание модели завершается с ошибкой"
        return

    try:
        clf = clf.fit(X, y)
    except Exception:
        assert False, "Обучение модели завершается с ошибкой"
        return

    assert isinstance(lossf(clf.w, X, y, 1e-3, 1e-3), float), "Функция потерь должна быть скалярной и иметь тип np.float"
    assert gradf(clf.w, X, y, 1e-3, 1e-3).shape == (100,), "Размерность градиента должна совпадать с числом параметров"
    assert gradf(clf.w, X, y, 1e-3, 1e-3).dtype == np.float, "Вектор градиента, должен состоять из элементов типа np.float"
    assert clf.predict(X).shape == (1000,), "Размер вектора предсказаний, должен совпадать с количеством объектов"
    assert np.min(clf.predict_proba(X)) >= 0, "Вероятности должны быть не меньше, чем 0"
    assert np.max(clf.predict_proba(X)) <= 1, "Вероятности должны быть не больше, чем 1"
    assert len(set(clf.predict(X))) == 2, "Метод предсказывает больше чем 2 класса на двух классовой задаче"
    print "End tests"

test_work()
