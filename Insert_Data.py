import os
os.environ['OPENCV_IO_ENABLE_JASPER']='True'
import cv2
import numpy as np
import math
import hamming
from scipy.fftpack import dct, idct

# Функция проверки числа на четность
def check_digit(digit):
    if digit % 2 == 0:
        return 0
    else:
        return 1

def int_r(num):
    num = int(num + (0.5 if num > 0 else -0.5))
    return num

def _iwt(array):
    output = np.zeros_like(array)
    nx, ny = array.shape
    x = nx // 2
    for j in range(ny):
        output[0:x,j] = (array[0::2,j] + array[1::2,j])//2
        output[x:nx,j] = array[0::2,j] - array[1::2,j]
    return output

def _iiwt(array):
    output = np.zeros_like(array)
    nx, ny = array.shape
    x = nx // 2
    for j in range(ny):
        output[0::2,j] = array[0:x,j] + (array[x:nx,j] + 1)//2
        output[1::2,j] = output[0::2,j] - array[x:nx,j]
    return output

def iwt2(array):
    return _iwt(_iwt(array.astype(int)).T).T

def iiwt2(array):
    return _iiwt(_iiwt(array.astype(int).T).T)

def iwt2N(array, N):
    arr = array
    for i in range(N):
        arr = iwt2(arr)
    return arr

def iiwt2N(array, N):
    arr = array
    for i in range(N):
        arr = iiwt2(arr)
    return arr


# Переводи значения ascii-символов в двоичный вид
def vec_bin_array(arr, m):
    """
    Arguments:
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain

    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int8's.
    """
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[...,bit_ix] = fetch_bit_func(strs).astype("int8")

    return ret


def bin_arr_to_int(bin_byte):
    res = int("".join(str(x) for x in bin_byte),2)
    return res



def paste_data_to_image(filename, output_format, file_key, secret_message, n_levels):
    # ============ SETTINGS ====================
    N = n_levels
    n_dubl = 1  # 1
    quality_image = 100  # степень сжатия .jpg от 0 до 100

    # ============ SETTINGS ====================
    original = cv2.imread(filename, cv2.IMREAD_COLOR)

    dim_x_old = original.shape[1]
    dim_y_old = original.shape[0]

    true_size = 0
    kratnoe = 2

    while true_size != 1:
        dim_x_ch = dim_x_old
        dim_y_ch = dim_y_old
        for i in range(n_levels+1):
            dim_x_ch = dim_x_ch / kratnoe
            if check_digit(dim_x_ch) != 0:
                dim_x_old += 1
                break
            dim_y_ch = dim_y_ch / kratnoe
            if check_digit(dim_y_ch) != 0:
                dim_y_old += 1
                break
            if i == n_levels:
                true_size = 1

    dim_x_new = dim_x_old
    dim_y_new = dim_y_old

    dim = (dim_x_new, dim_y_new)
    original = cv2.resize(original, dim, interpolation = cv2.INTER_AREA)


    original = np.array(original)
    original = np.uint8(original)

    original_blue = np.zeros((original.shape[0], original.shape[1]))
    original_green = np.zeros((original.shape[0], original.shape[1]))
    original_red = np.zeros((original.shape[0], original.shape[1]))

    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            original_blue[i][j] = original[i][j][0]
            original_green[i][j] = original[i][j][1]
            original_red[i][j] = original[i][j][2]

    # =================================================================
    coeffs2_blue = iwt2N(original_blue, N)
    coeffs2_green = iwt2N(original_green, N)
    coeffs2_red = iwt2N(original_red, N)
    # =================================================================

    d_x = int(dim_x_new/2)
    d_y = int(dim_y_new/2)

    #d_x = dim_x_new
    #d_y = dim_y_new

    LL1 = coeffs2_blue[0:d_x,0:d_y]
    LL2 = coeffs2_green[0:d_x,0:d_y]
    LL3 = coeffs2_red[0:d_x,0:d_y]

    LL = LL1
    LL = np.array(LL)

    LL_old1 = LL1
    LL_old2 = LL2
    LL_old3 = LL3

    # Считываем ключ из файла
    key = np.load(file_key)
    key = np.array(key, dtype=np.int32)

    # Рассчитываем максимальную длину сообщения, которую можно внедрить
    length_of_symbol_ASCII = 8
    length_of_symbol_UNICODE = 16

    # Выбор кодировки
    length_of_symbol = length_of_symbol_ASCII

    if LL.shape[0] > LL.shape[1]:
        LL_min = LL.shape[1]
    else:
        LL_min = LL.shape[0]

    if math.sqrt(key.shape[1]) > LL_min:
        max_side = LL_min
    else:
        max_side = int(math.sqrt(key.shape[1]))

    max_length_message = max_side * max_side / length_of_symbol / n_dubl / 1.5

    max_length_message = int(max_length_message)
    max_length_message = max_length_message - 2

    if output_format == "jpg":
        max_length_message = int(max_length_message * 1.0)

    #print("Максимальная длина внедряемого сообщения не должна превышать (в символах): ", max_length_message)

    # Вводим передаемое сообщение
    message = secret_message
    message += "@"

    length_of_message = len(message)
    #print("Длина введенного сообщения: ", length_of_message)
    #print("Встраивание информации в изображение...")

    if (len(message) > max_length_message):
        #print("Превышена максимальная длина сообщения")
        #print("Работа программы будет завершена с ошибкой.")
        #exit(-1)
        return -1

    dubl_message = []

    for c in message:
        for i in range(n_dubl):
            dubl_message.append(c)

    ascii_message = [ord(c) for c in dubl_message] # message
    ascii_message = np.asarray(ascii_message)



    bin_arr = vec_bin_array(ascii_message, length_of_symbol)

    # Длина блока кодирования
    CHUNK_LENGTH = length_of_symbol # 8

    # проверка длины блока
    assert not CHUNK_LENGTH % length_of_symbol, 'Длина блока должна быть кратна 8' # 8

    # вычисление контрольных бит
    CHECK_BITS = [i for i in range(1, CHUNK_LENGTH + 1) if not i & (i - 1)]

    bin_arr1 = []
    for i in bin_arr:
        buf = []
        buf_str = ""
        for j in i:
            buf_str = buf_str + str(j)
            #print("buf_str = ", buf_str)
        buf_str = hamming.encode_my(buf_str)
        #print("buf_str_cod = ", buf_str)
        for ii in buf_str:
            buf.append(int(ii))
        bin_arr1.append(buf)

    bin_arr = bin_arr1
    bin_arr = np.array(bin_arr)


    # Изменяем коэффициенты
    count = 0
    for byte in bin_arr:
        for bit in byte:
            check_dig = LL1[key[0][count]][key[1][count]]
            if check_digit(check_dig) == 1 and bit == 0:
                LL1[key[0][count]][key[1][count]] = LL1[key[0][count]][key[1][count]] + 1
            else:
                if check_digit(check_dig) == 0 and bit == 1:
                    LL1[key[0][count]][key[1][count]] = LL1[key[0][count]][key[1][count]] + 1
            count = count + 1


    LL_change1 = LL1

    count = 0
    for byte in bin_arr:
        for bit in byte:
            check_dig = LL2[key[0][count]][key[1][count]]
            if check_digit(check_dig) == 1 and bit == 0:
                LL2[key[0][count]][key[1][count]] = LL2[key[0][count]][key[1][count]] + 1
            else:
                if check_digit(check_dig) == 0 and bit == 1:
                    LL2[key[0][count]][key[1][count]] = LL2[key[0][count]][key[1][count]] + 1
            count = count + 1

    LL_change2 = LL2

    count = 0
    for byte in bin_arr:
        for bit in byte:
            check_dig = LL3[key[0][count]][key[1][count]]
            if check_digit(check_dig) == 1 and bit == 0:
                LL3[key[0][count]][key[1][count]] = LL3[key[0][count]][key[1][count]] + 1
                if check_digit(check_dig) == 0 and bit == 1:
                    LL3[key[0][count]][key[1][count]] = LL3[key[0][count]][key[1][count]] + 1
            count = count + 1

    LL_change3 = LL3

    # Выполняем обратное вейвлет-преобразование
    coeffs2_blue[0:d_x,0:d_y] = LL1 #
    coeffs2_green[0:d_x,0:d_y] = LL2 #
    coeffs2_red[0:d_x,0:d_y] = LL3 #

    original_change_blue = iiwt2N(coeffs2_blue, N)
    original_change_blue = np.array(original_change_blue)

    original_change_green = iiwt2N(coeffs2_green, N)
    original_change_green = np.array(original_change_green)

    original_change_red = iiwt2N(coeffs2_red, N)
    original_change_red = np.array(original_change_red)

    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            original[i][j][0] = int_r(original_change_blue[i][j])
            original[i][j][1] = int_r(original_change_green[i][j])
            original[i][j][2] = int_r(original_change_red[i][j])

    original = np.uint8(original)

    # Записываем изображение в файл
    filename_change = "Stego." + output_format
    cv2.imwrite(filename_change, original, [int(cv2.IMWRITE_JPEG_QUALITY), quality_image])



    # АЛГОРИТМ ИЗВЛЕЧЕНИЯ СТЕГАНОГРАФИИ ИЗ ИЗОБРАЖЕНИЯ ==============================================
    #print("Извлечение вложения из изображения...")
    original_change_read = cv2.imread(filename_change, cv2.IMREAD_COLOR)
    original_change_read = np.array(original_change_read)

    for i in range(original_change_read.shape[0]):
        for j in range(original_change_read.shape[1]):
            for c in range(original_change_read.shape[2]):
                original_change_read[i][j][c] = int_r(original_change_read[i][j][c])

    original_change_read = np.uint8(original_change_read)

    original_blue_change = np.zeros((original_change_read.shape[0], original_change_read.shape[1]))
    original_green_change = np.zeros((original_change_read.shape[0], original_change_read.shape[1]))
    original_red_change = np.zeros((original_change_read.shape[0], original_change_read.shape[1]))

    for i in range(original_change_read.shape[0]):
        for j in range(original_change_read.shape[1]):
            original_blue_change[i][j] = original_change_read[i][j][0]
            original_green_change[i][j] = original_change_read[i][j][1]
            original_red_change[i][j] = original_change_read[i][j][2]

    original_blue_change = np.uint8(original_blue_change)
    original_green_change = np.uint8(original_green_change)
    original_red_change = np.uint8(original_red_change)

    coeffs2_blue = iwt2N(original_blue_change, N)
    coeffs2_green = iwt2N(original_green_change, N)
    coeffs2_red = iwt2N(original_red_change, N)

    LL1 = coeffs2_blue[0:d_x,0:d_y] #
    LL2 = coeffs2_green[0:d_x,0:d_y] #
    LL3 = coeffs2_red[0:d_x,0:d_y] #

    LL = LL1
    LL = np.array(LL)

    for i in range(LL.shape[0]):
        for j in range(LL.shape[1]):
            LL1[i][j] = int(int_r(LL1[i][j]))
            LL2[i][j] = int(int_r(LL2[i][j]))
            LL3[i][j] = int(int_r(LL3[i][j]))

    # Извлечение сообщения из изображения
    stop_char = '@'
    current_char = ""

    length_of_symbol = length_of_symbol * 1.5
    length_of_symbol = int(length_of_symbol)

    errors1 = 0
    errors2 = 0
    errors3 = 0

    message_1 = ""
    count = 0
    count_sym = 0
    count_sym_1 = 0
    stop_count_real = 0
    stop_count_ideal = int(n_dubl/3)
    if stop_count_ideal < 1:
        stop_count_ideal = 1

    while current_char != ord(stop_char) and count_sym_1 + 1 < max_length_message:
        byte = []
        for i in range(length_of_symbol):
            check_dig = LL1[key[0][count]][key[1][count]]
            if LL1[key[0][count]][key[1][count]] == LL_change1[key[0][count]][key[1][count]] == LL_old1[key[0][count]][key[1][count]]:
                print("", end="")
            else:
                if LL1[key[0][count]][key[1][count]] != LL_change1[key[0][count]][key[1][count]]:
                    errors1 += 1
            if check_digit(check_dig) == 1:
                byte.append(1)
            else:
                if check_digit(check_dig) == 0:
                    byte.append(0)
            count = count + 1
        bin_arr = byte
        buf = []
        buf_str = ""
        for j in bin_arr:
            buf_str = buf_str + str(j)
        buf_str = hamming.decode_my(buf_str)
        for ii in buf_str:
            buf.append(int(ii))
        bin_arr = buf
        byte = bin_arr
        current_char = bin_arr_to_int(byte)
        message_1 = message_1 + chr(current_char)
        if count_sym_1 == n_dubl:
            stop_count_real = 0
            count_sym = 0
            count_sym_1 = 0
        if ( current_char == ord(stop_char)):
            stop_count_real += 1
        count_sym += 1
        count_sym_1 += 1

    current_char = ""
    message_2 = ""
    count = 0
    count_sym = 0
    count_sym_1 = 0
    stop_count_real = 0
    stop_count_ideal = int(n_dubl/3)
    if stop_count_ideal < 1:
        stop_count_ideal = 1

    while current_char != ord(stop_char) and count_sym_1 + 1 < max_length_message:
        byte = []
        for i in range(length_of_symbol):
            check_dig = LL2[key[0][count]][key[1][count]]
            if LL_old2[key[0][count]][key[1][count]] == LL_change2[key[0][count]][key[1][count]] == LL2[key[0][count]][key[1][count]]:
                print("", end="")
            else:
                if LL2[key[0][count]][key[1][count]] != LL_change2[key[0][count]][key[1][count]]:
                    errors2 += 1
            if check_digit(check_dig) == 1:
                byte.append(1)
            else:
                if check_digit(check_dig) == 0:
                    byte.append(0)
            count = count + 1
        bin_arr = byte
        buf = []
        buf_str = ""
        for j in bin_arr:
            buf_str = buf_str + str(j)
        buf_str = hamming.decode_my(buf_str)
        for ii in buf_str:
            buf.append(int(ii))
        bin_arr = buf
        byte = bin_arr
        current_char = bin_arr_to_int(byte)
        message_2 = message_2 + chr(current_char)
        if count_sym == n_dubl:
            stop_count_real = 0
            count_sym = 0
            count_sym_1 = 0
        if ( current_char == ord(stop_char)):
            stop_count_real += 1
        count_sym += 1
        count_sym_1 += 1

    current_char = ""
    message_3 = ""
    count = 0
    count_sym = 0
    count_sym_1 = 0
    stop_count_real = 0
    stop_count_ideal = int(n_dubl/3)
    if stop_count_ideal < 1:
        stop_count_ideal = 1

    while current_char != ord(stop_char) and count_sym_1 + 1 < max_length_message:
        byte = []
        for i in range(length_of_symbol):
            check_dig = LL3[key[0][count]][key[1][count]]
            if LL3[key[0][count]][key[1][count]] == LL_change3[key[0][count]][key[1][count]] == LL_old3[key[0][count]][key[1][count]]:
                print("", end="")
            else:
                if LL3[key[0][count]][key[1][count]] != LL_change3[key[0][count]][key[1][count]]:
                    errors3 += 1
            if check_digit(check_dig) == 1:
                byte.append(1)
            else:
                if check_digit(check_dig) == 0:
                    byte.append(0)
            count = count + 1
        bin_arr = byte
        buf = []
        buf_str = ""
        for j in bin_arr:
            buf_str = buf_str + str(j)
        buf_str = hamming.decode_my(buf_str)
        for ii in buf_str:
            buf.append(int(ii))
        bin_arr = buf
        byte = bin_arr
        current_char = bin_arr_to_int(byte)
        message_3 = message_3 + chr(current_char)
        if count_sym == n_dubl:
            stop_count_real = 0
            count_sym = 0
            count_sym_1 = 0
        if (current_char == ord(stop_char)):
            stop_count_real += 1
        count_sym += 1
        count_sym_1 += 1

    real_message_1 = ""
    real_message_2 = ""
    real_message_3 = ""

    freq = np.full((n_dubl), -1)

    # Анализ сообщения:
    for i in range (0,len(message_1),n_dubl):
        arr_sym = []
        for j in range (i,i+n_dubl):
            if j < len(message_1):
                arr_sym.append(message_1[j])
        # Поиск по мажоритарному принципу наиболее вероятный символ из n_dubl:
        for ii in range(n_dubl):
            for jj in range(n_dubl):
                if jj < len(message_1) and ii < len(message_1) and jj < len(arr_sym) and ii < len(arr_sym):
                    if arr_sym[ii] == arr_sym[jj]:
                        freq[ii] += 1
        maximum = 0
        max_id = 0
        for g in range(0,n_dubl):
            if g < len(arr_sym):
                if freq[g] > maximum:
                    maximum = freq[g]
        for gg in range(0,n_dubl):
            if gg < len(arr_sym):
                if freq[gg] == maximum:
                    max_id = gg
                    break
        real_message_1 += arr_sym[max_id]

    freq = np.full((n_dubl), -1)

    for i in range (0,len(message_2),n_dubl):
        arr_sym = []
        for j in range (i,i+n_dubl):
            if j < len(message_2):
                arr_sym.append(message_2[j])
        for ii in range(n_dubl):
            for jj in range(n_dubl):
                if jj < len(message_2) and ii < len(message_2) and jj < len(arr_sym) and ii < len(arr_sym):
                    if arr_sym[ii] == arr_sym[jj]:
                        freq[ii] += 1
        maximum = 0
        max_id = 0
        for g in range(n_dubl):
            if g < len(arr_sym):
                if freq[g] > maximum:
                    maximum = freq[g]
        for gg in range(n_dubl):
            if gg < len(arr_sym):
                if freq[gg] == maximum:
                    max_id = gg
                    break
        real_message_2 += arr_sym[max_id]

    freq = np.full((n_dubl), -1)

    for i in range (0,len(message_3),n_dubl):
        arr_sym = []
        for j in range (i,i+n_dubl):
            if j < len(message_3):
                arr_sym.append(message_3[j])
        for ii in range(n_dubl):
            for jj in range(n_dubl):
                if jj < len(message_3) and ii < len(message_3) and jj < len(arr_sym) and ii < len(arr_sym):
                    if arr_sym[ii] == arr_sym[jj]:
                        freq[ii] += 1
        maximum = 0
        max_id = 0
        for g in range(n_dubl):
            if g < len(arr_sym):
                if freq[g] > maximum:
                    maximum = freq[g]
        for gg in range(n_dubl):
            if gg < len(arr_sym):
                if freq[gg] == maximum:
                    max_id = gg
                    break
        real_message_3 += arr_sym[max_id]

    len_real_message_max = 0

    if len(real_message_1) > len_real_message_max:
        len_real_message_max = len(real_message_1)

    if len(real_message_2) > len_real_message_max:
        len_real_message_max = len(real_message_2)

    if len(real_message_3) > len_real_message_max:
        len_real_message_max = len(real_message_3)

    r_m_1 = ""
    r_m_2 = ""
    r_m_3 = ""

    for i in range(0,len(real_message_1)):
        r_m_1 += real_message_1[i]
    for i in range(len(real_message_1),len_real_message_max):
        r_m_1 += "?"

    for i in range(0,len(real_message_2)):
        r_m_2 += real_message_2[i]
    for i in range(len(real_message_2),len_real_message_max):
        r_m_2 += "?"

    for i in range(0,len(real_message_3)):
        r_m_3 += real_message_3[i]
    for i in range(len(real_message_3),len_real_message_max):
        r_m_3 += "?"

    real_message_1 = r_m_1
    real_message_2 = r_m_2
    real_message_3 = r_m_3

    real_message = ""
    for i in range(0,len_real_message_max):
        if real_message_1[i] == real_message_2[i] == real_message_3[i]:
            real_message += real_message_1[i]
            continue
        if real_message_1[i] == real_message_2[i]:
            real_message += real_message_1[i]
            continue
        if real_message_1[i] == real_message_3[i]:
            real_message += real_message_1[i]
            continue
        if real_message_2[i] == real_message_3[i]:
            real_message += real_message_2[i]
            continue
        # Если все разные, то принимать за верный символ из канала канала BLUE
        real_message += real_message_1[i]
    #print("\n\n------------------------------")
    #print("Результаты встраивания:\n")

    #print("========================================")
    #print("Message_in_BLUE_channel = ", real_message_1[:-1])
    #print("Message_in_GREEN_channel = ", real_message_2[:-1])
    #print("Message_in_RED_channel = ", real_message_3[:-1])
    #print("========================================")
    #print("========================================")
    #print("Number of errors: ")
    #print("Errors_BLUE = ", errors1)
    #print("Errors_GREEN = ", errors2)
    #print("Errors_RED = ", errors3)
    #print("========================================\n")


    real_len_mess = 0
    for ch in real_message:
        if ch == '@':
            break
        real_len_mess += 1

    real_message = real_message[:real_len_mess]

    if real_message == message[:-1]:
        #print("Сообщение [" + real_message + "] успешно внедрено в изображение: " + filename_change)
        print("")
    else:
        #print("Внедреенное сообщение [" + real_message + "] не совпадает с исходным [" + message[:-1] + "]")
        #print("Попробуйте другое изображение или измените сообщение. Также проверьте все условия генерации ключа.")
        return -1

    #print("Работа программы успешно завершена.")

    return 0
    # ===

'''
filename = "1.jpg"
output_format = "bmp"
file_key = "key.npy"
secret_message = "Hello, world!"
result = paste_data_to_image(filename, output_format, file_key, secret_message)
if result != 0:
    print("Встраивание завершилось с ошибкой")
else:
    print("Встраивание завершилось с успешно")
'''