import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import os
from datetime import datetime

class LandingZoneDetector:
    def __init__(self, min_contour_area=500, target_aspect_ratio=1.0, tolerance=0.2):
        '''
        Инициализация детектора посадочной площадки
        
        Параметры:
        - min_contour_area: минимальная площадь контура для рассмотрения (в пикселях)
        - target_aspect_ratio: ожидаемое соотношение сторон площадки (1.0 для квадрата)
        - tolerance: допустимое отклонение от ожидаемого соотношения сторон
        (если tolerance=0.2, то  значение может быть в target_aspect_ratio 1.2 или 0.8)
        '''
        self.min_contour_area = min_contour_area
        self.target_aspect_ratio = target_aspect_ratio
        self.tolerance = tolerance
        
    def preprocess_frame(self, frame):
        '''
        Шаг 1: Предварительная обработка кадра
        - Конвертация в оттенки серого
        - Гауссово размытие для уменьшения шума
        
        Параметры:
        - frame: входной кадр в формате 
        
        Возвращает:
        - Обработанное изображение в оттенках серого
        '''

        '''
        Преобразование цветного изображения из формата BGR в оттенки серого
        использует формулу: gray = 0.299 * R + 0.587 * G + 0.114 * B
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        '''
        Выполняет размытие изображения с помощью гауссова фильтра
        - gray: кадр чернобелый
        - (9, 9): Размер ядра (kernel) гауссова фильтра. Здесь используется квадратное ядро 5х5 пикселей.
            Чем больше ядро, тем сильнее размытие.
        - 0: Стандартное отклонение (σ) по оси X. Если указать 0, OpenCV автоматически вычислит σ на основе размера ядра.
        '''
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        return blurred
    
    def detect_edges(self, image):
        '''
        Шаг 2: Обнаружение границ с помощью оператора Кэнни
        Использует алгоритм Canny Edge Detection для обнаружения границ на изображении
        Параметры:
        - image: изображение в оттенках серого
        
        Возвращает:
        - Бинарное изображение с границами
        '''
        '''Вычисляет медиану значений пикселей изображения image (в оттенках серого).'''
        v = np.median(image)
        '''Расчёт нижнего и верхнего порогов'''
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))

        '''
        cv2.Canny применяет алгоритм обнаружения границ:
            Сглаживание изображения (гауссово размытие, если оно не было выполнено ранее).
            Поиск градиентов (оператор Собеля для вычисления производных по X и Y).
            Подавление не-максимумов — оставляет только тонкие линии границ.
            Отбрасывает слабые границы, не связанные с сильными.
        Бинарное изображение (edged), где:
            255 (белый) — пиксели, принадлежащие границам.
            0 (чёрный) — фон.
        '''
        edged = cv2.Canny(image, lower, upper)
        return edged
    
    def find_contours(self, edged_image):
        '''
        Шаг 3: Поиск контуров на изображении с границами
        Использует OpenCV для поиска контуров на бинарном изображении с границами (полученном после детектора Кэнни).
        Параметры:
        - edged_image: бинарное изображение с границами
        
        Возвращает:
        - Список найденных контуров
        '''



        '''
        cv2.findContours() - список контуров (каждый контур — это массив точек (x, y))
        edged_image.copy()
            Создаётся копия изображения с границами, чтобы исходное не изменялось.
            Важно: findContours может модифицировать входное изображение, поэтому используется .copy().
        cv2.RETR_EXTERNAL
            Режим извлечения контуров: возвращает только внешние (крайние) контуры.
            Пример: для объекта с "дыркой" (например, буква "О") вернётся только внешний контур, а внутренний (границы дырки) будет проигнорирован.
        cv2.CHAIN_APPROX_SIMPLE
            Метод аппроксимации контуров: сжимает контур, удаляя избыточные точки.
            Пример: для прямой линии сохраняются только начальная и конечная точки.
        '''
        contours = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



        '''для состыковки версий(для того чтобы не проверять какие значения возвращает cv2.findContours)'''
        contours = imutils.grab_contours(contours)


        '''
        Структура контуров:
            Каждый контур — это массив точек формата [[[x1, y1]], [[x2, y2]], ...] (numpy-массив).
        Пример контура:
            Для прямоугольника контур будет состоять из 4 точек (углов).
            Для круга — из множества точек, аппроксимирующих окружность.
        '''
        return contours
    
    def filter_contours(self, contours):
        '''
         Шаг 4: Фильтрация контуров по площади и форме
        
        Параметры:
        - contours: список контуров
        
        Возвращает:
        - Отфильтрованный список контуров, соответствующих критериям посадочной площадки
        '''
        filtered_contours = []
        
        for contour in contours:

            '''
            Вычисляет площадь контура в пикселях.
            '''
            area = cv2.contourArea(contour)
            '''
            Контуры с площадью меньше min_contour_area отбрасываются как слишком мелкие (шум или артефакты).
            '''
            if area < self.min_contour_area:
                continue
                

            '''
            Аппроксимация формы контура
            Результат: Контур превращается в многоугольник с меньшим числом вершин (например, квадрат → 4 точки).
            '''

            '''
            perimeter = cv2.arcLength(contour, True)
            - cv2.arcLength(contour, True) вычисляет периметр (длину) контура. Параметр True означает, что контур замкнутый.
            '''
            perimeter = cv2.arcLength(contour, True)


            '''
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            - cv2.approxPolyDP() упрощает контур, сохраняя его общую форму:
            0.02 * perimeter — максимальное отклонение от исходного контура (2% от периметра).
            '''
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            

            '''
            Фильтрация по соотношению сторон
            cv2.boundingRect(approx) вычисляет ограничивающий прямоугольник для аппроксимированного контура.
                Возвращает координаты (x, y) верхнего левого угла, ширину w и высоту h.
            aspect_ratio — соотношение сторон прямоугольника (width / height).
                Проверяется, насколько это соотношение отличается от target_aspect_ratio 
                (например, 1.0 для квадрата) с учётом допуска tolerance (например, ±0.2).
            '''
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            

            '''отброс не нужных контуров'''
            if abs(aspect_ratio - self.target_aspect_ratio) > self.tolerance:
                continue
            

            '''Сохранение подходящих контуров'''
            filtered_contours.append(contour)
            
        return filtered_contours
    
    def calculate_contour_center_and_orientation(self, contour):
        '''
        Шаг 5: Вычисление центра масс и ориентации контура
        
        Параметры:
        - contour: входной контур
        
        Возвращает:
        - Координаты центра (cx, cy) и угол ориентации в градусах
        '''


        '''
        Вычисление моментов контура
        
        cv2.moments() рассчитывает моменты контура — набор статистических характеристик, описывающих его форму и расположение.
            Возвращает словарь M с различными типами моментов:
            Простые моменты: m00, m10, m01 (используются для центра масс).
            Центральные моменты: mu20, mu02, mu11 (используются для ориентации).
        '''
        M = cv2.moments(contour)
        

        '''
        Расчёт центра масс (центроида)

        m00 — площадь контура (в пикселях). Если равна 0, контур некорректен.
        Формулы центра:
            cx = m10 / m00 — координата X центра масс.
            cy = m01 / m00 — координата Y центра масс.
            Результат округляется до целых пикселей (int).
        '''
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
            


        '''
        Расчёт угла ориентации

        Используются центральные моменты 2-го порядка:
            mu20 — момент инерции относительно оси X.
            mu02 — момент инерции относительно оси Y.
            mu11 — корреляционный момент.
        Формула угла:
            theta = 0.5 * arctan2(2 * mu11, (mu20 - mu02)) — угол в радианах.
            np.degrees() преобразует угол в градусы.
            Физический смысл: Ось, относительно которой контур наиболее "вытянут".

        '''
        mu20 = M['mu20']
        mu02 = M['mu02']
        mu11 = M['mu11']
        
        theta = 0.5 * np.arctan2(2 * mu11, (mu20 - mu02))
        angle = np.degrees(theta)
        
        return (cx, cy), angle
    

    def _debug_visualization(self, original, processed, edged, contour, center, angle, frame_count):
        '''
        Вспомогательный метод для визуализации промежуточных результатов
        с сохранением в папку debug_output
        '''

        '''
        Создаем папку для отладочных изображений, если её нет

        Создаёт папку debug_output, если она не существует.
        Параметр exist_ok=True предотвращает ошибку, если папка уже есть.
        '''
        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)

        '''
        Создаем копии изображений для визуализации

        Конвертация в цветное изображение:
            processed (чёрно-белое размытое) и edged (границы Кэнни) преобразуются из GRAY в BGR для цветной визуализации.
        Копия оригинального кадра (vis_contour) для рисования результатов.
        '''
        vis_processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        vis_edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
        vis_contour = original.copy()
        
        '''
        Рисуем контур, центр и линию ориентации
        
        Контур:
            Рисуется зелёным цветом ((0, 255, 0)) с толщиной 2 пикселя.
        Центр масс:
            Красная точка ((0, 0, 255)) диаметром 5 пикселей.
        '''
        cv2.drawContours(vis_contour, [contour], -1, (0, 255, 0), 2)
        cv2.circle(vis_contour, center, 5, (0, 0, 255), -1)
        

        '''
        Рисование оси ориентации

        Вычисление конца линии:
            Линия длиной 50 пикселей от центра под углом angle.
        Рисование:
            Синяя линия ((255, 0, 0)) с толщиной 2 пикселя.
        '''
        x1 = center[0] + 50 * np.cos(np.radians(angle))
        y1 = center[1] + 50 * np.sin(np.radians(angle))
        cv2.line(vis_contour, center, (int(x1), int(y1)), (255, 0, 0), 2)
        
        '''
        Создание составного изображения (subplots)

        Инициализация фигуры Matplotlib размером 15x10 дюймов.
        '''
        plt.figure(figsize=(15, 10))
        
        '''
        Оригинальный кадр
        
        Конвертация из BGR (OpenCV) в RGB (Matplotlib).
        '''
        plt.subplot(221)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Original Frame")
        
        '''Предобработанное изображение'''
        plt.subplot(222)
        plt.imshow(cv2.cvtColor(vis_processed, cv2.COLOR_BGR2RGB))
        plt.title("Processed (Grayscale + Blur)")
        
        '''Границы Кэнни:'''
        plt.subplot(223)
        plt.imshow(cv2.cvtColor(vis_edged, cv2.COLOR_BGR2RGB))
        plt.title("Edge Detection")
        
        '''Результат детекции:'''
        plt.subplot(224)
        plt.imshow(cv2.cvtColor(vis_contour, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Landing Zone (Angle: {angle:.1f}°)")
        

        '''
        Сохранение и завершение
        
        tight_layout(): Автоматическая корректировка отступов между подграфиками.
        Имя файла:
            Включает номер кадра (frame_count) и временную метку.
        Пример: debug_0042_20230815_143022.png.
        Сохранение:
            В формате PNG с разрешением 100 DPI.
            plt.close(): Освобождение памяти, занятой фигурой.
        '''
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_filename = os.path.join(debug_dir, f"debug_{frame_count:04d}_{timestamp}.png")
        plt.savefig(debug_filename, dpi=100, bbox_inches='tight')
        plt.close() 

    def detect_landing_zone(self, frame, debug=False, frame_count=0):
        '''
        Основной метод для обнаружения посадочной площадки
        
        Параметры:
        - frame: входной кадр
        - debug: флаг отладки (True/False)
        - frame_count: номер кадра (для сохранения отладочных изображений)
        
        Возвращает:
        - ((cx, cy), angle, detected) при успешном обнаружении
        - ((0, 0), 0, False) при ошибке
        '''
        try:
            ''' Проверка входного кадра '''
            if frame is None:
                return (0, 0), 0, False
                
            '''Шаг 1: Предварительная обработка'''
            processed = self.preprocess_frame(frame)

            '''Шаг 2: Обнаружение границ'''
            edged = self.detect_edges(processed)

            '''Шаг 3: Поиск контуров'''
            contours = self.find_contours(edged)

            '''Шаг 4: Фильтрация контуров'''
            filtered_contours = self.filter_contours(contours)
            
            '''Если контуры не найдены'''
            if not filtered_contours:
                return (0, 0), 0, False

            '''Выбираем наибольший контур'''     
            largest_contour = max(filtered_contours, key=cv2.contourArea)

            '''Шаг 5: Вычисление центра и ориентации'''
            (cx, cy), angle = self.calculate_contour_center_and_orientation(largest_contour)
            
            if debug:
                self._debug_visualization(frame, processed, edged, largest_contour, (cx, cy), angle, frame_count)
                
            return (cx, cy), angle, True
            
        except Exception as e:
            print(f"Ошибка в detect_landing_zone: {str(e)}")
            return (0, 0), 0, False

def main():
    '''Инициализация детектора'''
    detector = LandingZoneDetector(
        min_contour_area=1000,
        target_aspect_ratio=1.0,
        tolerance=0.5
    )
    
    '''
    Инициализация видеопотока(возвращает объект VideoCapture )
    Создает объект для чтения видео из файла 
    Приемры квадрат: test_video_5.mp4
    Приемры круг: test_video_9.mp4
    Приемры прямоугольник: test_video_11.mp4
    ''' 
    cap = cv2.VideoCapture("test_video_11.mp4")
    
    '''Проверка успешного открытия камеры'''
    if not cap.isOpened():
        print("Ошибка: не удалось получить доступ к камере")
        print("Попробуйте следующее:")
        print("1. Убедитесь, что камера не используется другим приложением")
        print("2. Проверьте разрешения для камеры в системных настройках")
        print("3. Для macOS выполните в терминале: tccutil reset Camera")
        return
    
    
    '''Получаем параметры видео для сохранения'''
    '''Получаем ширину кадра 720'''
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ''' Получаем высоту кадра 1280'''
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ''' Получаем частоту кадров 29'''
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    '''
    Создаем объект для записи видео
    Создаёт FourCC (Four Character Code) — 4-байтовый идентификатор видеокодека.
    '''
    fourcc = cv2.VideoWriter_fourcc(*'h264')


    '''
    Создаёт объект VideoWriter для записи видеофайла с параметрами:
    - 'output_video.mp4': Имя выходного файла
    - fourcc: Кодек, полученный на предыдущем шаге (в нашем случае 'mp4').
    - fps: Частота кадров. Должна совпадать с FPS исходного видео.
    - (frame_width, frame_height): Разрешение видео в пикселях (ширина и высота). Должно совпадать с размером кадров.
    '''
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    
    while True:

        '''
        Используется для чтения очередного кадра из видеофайла 
        - ret (bool): 
            True — если кадр успешно прочитан.
            False — если кадр не прочитан(конец видео или ошибка).
        - frame (numpy.ndarray):
            Массив NumPy, содержащий изображение кадра в формате BGR (если ret == True).
            None — если кадр не прочитан (ret == False).
        '''
        ret, frame = cap.read()
        if not ret:
            break
            
        try:
            '''
            detect_landing_zone выполняет обнаружение посадочной площадки на переданном кадре изображения
            и возвращает её координаты, угол ориентации и статус обнаружения.
            Входные параметры:
            - frame (numpy.ndarray):
                Массив NumPy, содержащий изображение кадра в формате BGR (если ret == True).
                None — если кадр не прочитан (ret == False).
            - debug=True — включение режима отладки:
                При True сохраняет промежуточные результаты обработки в папку debug_output.
                При False отладка отключена (только возврат результатов).
            - framecount: Номер кадра (используется для именования отладочных файлов)
            Выходные параметры:
            - (cx, cy) — координаты центра площадки в пикселях:
                cx — координата по горизонтали (ось X).
                cy — координата по вертикали (ось Y).
            - angle — угол ориентации площадки в градусах (°):
                0° — горизонтальное положение.
                Положительные/отрицательные значения — поворот по/против часовой стрелки.
            - detected — флаг успешности обнаружения (True/False).
            '''
            (cx, cy), angle, detected = detector.detect_landing_zone(frame, debug=False, frame_count=0)
            
            '''Отображение результатов на кадре'''
            if detected:
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
                cv2.putText(frame, f"Angle: {angle:.1f}°", (cx + 15, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 125, 0), 2)
            
            '''Запись кадра в выходной файл'''
            out.write(frame)
            '''Показ кадра в окне'''
            cv2.imshow("Landing Zone Detection", frame)
            
            frame_count += 1
            
            '''выход на q'''
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Ошибка обработки кадра: {str(e)}")
            continue

    '''Закрытие видеофайла'''
    cap.release()

    '''Закрытие файла записи'''
    out.release()
    '''Закрытие окон OpenCV'''
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

