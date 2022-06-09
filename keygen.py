import numpy as np

# Ввод предупрждения об ограничениях на размерность ключа
print("!!! Внимание, ширину и высоту ключа должны быть МЕНЬШЕ размерности LL-коэффициентов,в которые будет осуществляться встраивание информации !!!")
print("Примечание: при 1-м уровне вейвлет-преобразования LL_h==Image_h/2, LL_w==Image_w/2, при 2-м уровне LL_h==Image_h/4, LL_w==Image_w/4 и т.д.")
high_LL_min = input("Введите минимальную высоту ключа: ")
high_LL_min = int(high_LL_min)
width_LL_min = input("Введите минимальную ширину ключа: ")
width_LL_min = int(width_LL_min)

# Уставнока размерность ключа
key = np.zeros((2,high_LL_min * width_LL_min))

# Алгоритм формирования ключа
jj = 0
ii = 0
count = 0

for j in range(0,high_LL_min * width_LL_min):
    key[0][j] = ii
    key[1][j] = jj
    jj += 1
    count += 1
    if jj == width_LL_min:
        jj = 0
    if count == width_LL_min:
        ii += 1
        count = 0


# Сохранение ключа в файл
key_file_name = 'key'
np.save(key_file_name, key)

print("Работа программы успешно завершена. Сформированный ключ находится в файле: " + key_file_name + ".npy")