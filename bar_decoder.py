from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


def collect_barcode_data(image):
    img = Image.fromarray(image)
    #img.show()
    width, height = img.size
    max_size = (width, 128)
    img.thumbnail(max_size)
    basewidth = 4 * width
    img = img.resize((basewidth, height), Image.ANTIALIAS)
    hor_line_bw = img.crop((0, int(height / 2), basewidth, int(height / 2) + 1)).convert('L')
    hor_data = np.asarray(hor_line_bw, dtype="int32")[0]

    hor_data = 255 - hor_data
    #hor_data = hor_data[12:len(hor_data)]
    return hor_data


def find_type_and_bit_width(hor_data):
    avg = np.average(hor_data)
    #plt.plot(hor_data)
    #plt.show()
    pos1, pos2 = -1, -1
    strips = ""
    bit_width_code128 = ""
    bit_width_ean13 = ""
    for p in range(hor_data.size - 2):
        if hor_data[p] < avg and hor_data[p + 1] > avg:
            strips += "1"
            if pos1 == -1:
                pos1 = p
                continue
            if strips == "101":
                pos2 = p
                bit_width_code128 = int((pos2 - pos1) / 3)
                if bit_width_code128 == 0:
                    bit_width_code128 += 1
            if len(strips) == 31:
                pos2 = p
                bit_width_ean13 = int((pos2 - pos1) / 48)
        if hor_data[p] > avg and hor_data[p + 1] < avg and strips != "":
            strips += "0"
    return bit_width_code128, bit_width_ean13, pos1


def get_code(hor_data, bit_width, pos1):
    avg = np.average(hor_data)
    bits = ""
    for p in range(hor_data.size - 2):
        if hor_data[p] > avg and hor_data[p + 1] < avg and p >= pos1:
            interval = p - pos1
            cnt = interval / bit_width
            strip_bits = int(round(cnt))
            if strip_bits == 0 and cnt != 0:
                strip_bits = 1
            elif strip_bits > 4:
                strip_bits = 4
            bits += "1" * strip_bits
            pos1 = p
        if hor_data[p] < avg and hor_data[p + 1] > avg and p >= pos1:
            interval = p - pos1
            cnt = interval / bit_width
            strip_bits = int(round(cnt))
            if strip_bits == 0 and cnt != 0:
                strip_bits = 1
            elif strip_bits > 4:
                strip_bits = 4
            bits += "0" * strip_bits
            pos1 = p
    return bits


def encode_characters(type):
    CODE128_CHART = """
            0	_	_	00	32	S	11011001100	212222
            1	!	!	01	33	!	11001101100	222122
            2	"	"	02	34	"	11001100110	222221
            3	#	#	03	35	#	10010011000	121223
            4	$	$	04	36	$	10010001100	121322
            5	%	%	05	37	%	10001001100	131222
            6	&	&	06	38	&	10011001000	122213
            7	'	'	07	39	'	10011000100	122312
            8	(	(	08	40	(	10001100100	132212
            9	)	)	09	41	)	11001001000	221213
            10	*	*	10	42	*	11001000100	221312
            11	+	+	11	43	+	11000100100	231212
            12	,	,	12	44	,	10110011100	112232
            13	-	-	13	45	-	10011011100	122132
            14	.	.	14	46	.	10011001110	122231
            15	/	/	15	47	/	10111001100	113222
            16	0	0	16	48	0	10011101100	123122
            17	1	1	17	49	1	10011100110	123221
            18	2	2	18	50	2	11001110010	223211
            19	3	3	19	51	3	11001011100	221132
            20	4	4	20	52	4	11001001110	221231
            21	5	5	21	53	5	11011100100	213212
            22	6	6	22	54	6	11001110100	223112
            23	7	7	23	55	7	11101101110	312131
            24	8	8	24	56	8	11101001100	311222
            25	9	9	25	57	9	11100101100	321122
            26	:	:	26	58	:	11100100110	321221
            27	;	;	27	59	;	11101100100	312212
            28	<	<	28	60	<	11100110100	322112
            29	=	=	29	61	=	11100110010	322211
            30	>	>	30	62	>	11011011000	212123
            31	?	?	31	63	?	11011000110	212321
            32	@	@	32	64	@	11000110110	232121
            33	A	A	33	65	A	10100011000	111323
            34	B	B	34	66	B	10001011000	131123
            35	C	C	35	67	C	10001000110	131321
            36	D	D	36	68	D	10110001000	112313
            37	E	E	37	69	E	10001101000	132113
            38	F	F	38	70	F	10001100010	132311
            39	G	G	39	71	G	11010001000	211313
            40	H	H	40	72	H	11000101000	231113
            41	I	I	41	73	I	11000100010	231311
            42	J	J	42	74	J	10110111000	112133
            43	K	K	43	75	K	10110001110	112331
            44	L	L	44	76	L	10001101110	132131
            45	M	M	45	77	M	10111011000	113123
            46	N	N	46	78	N	10111000110	113321
            47	O	O	47	79	O	10001110110	133121
            48	P	P	48	80	P	11101110110	313121
            49	Q	Q	49	81	Q	11010001110	211331
            50	R	R	50	82	R	11000101110	231131
            51	S	S	51	83	S	11011101000	213113
            52	T	T	52	84	T	11011100010	213311
            53	U	U	53	85	U	11011101110	213131
            54	V	V	54	86	V	11101011000	311123
            55	W	W	55	87	W	11101000110	311321
            56	X	X	56	88	X	11100010110	331121
            57	Y	Y	57	89	Y	11101101000	312113
            58	Z	Z	58	90	Z	11101100010	312311
            59	[	[	59	91	[	11100011010	332111
            60	\	\	60	92	\	11101111010	314111
            61	]	]	61	93	]	11001000010	221411
            62	^	^	62	94	^	11110001010	431111
            63	_	_	63	95	_	10100110000	111224
            64	NUL	`	64	96	`	10100001100	111422
            65	SOH	a	65	97	a	10010110000	121124
            66	STX	b	66	98	b	10010000110	121421
            67	ETX	c	67	99	c	10000101100	141122
            68	EOT	d	68	100	d	10000100110	141221
            69	ENQ	e	69	101	e	10110010000	112214
            70	ACK	f	70	102	f	10110000100	112412
            71	BEL	g	71	103	g	10011010000	122114
            72	BS	h	72	104	h	10011000010	122411
            73	HT	i	73	105	i	10000110100	142112
            74	LF	j	74	106	j	10000110010	142211
            75	VT	k	75	107	k	11000010010	241211
            76	FF	l	76	108	l	11001010000	221114
            77	CR	m	77	109	m	11110111010	413111
            78	SO	n	78	110	n	11000010100	241112
            79	SI	o	79	111	o	10001111010	134111
            80	DLE	p	80	112	p	10100111100	111242
            81	DC1	q	81	113	q	10010111100	121142
            82	DC2	r	82	114	r	10010011110	121241
            83	DC3	s	83	115	s	10111100100	114212
            84	DC4	t	84	116	t	10011110100	124112
            85	NAK	u	85	117	u	10011110010	124211
            86	SYN	v	86	118	v	11110100100	411212
            87	ETB	w	87	119	w	11110010100	421112
            88	CAN	x	88	120	x	11110010010	421211
            89	EM	y	89	121	y	11011011110	212141
            90	SUB	z	90	122	z	11011110110	214121
            91	ESC	{	91	123	{	11110110110	412121
            92	FS	|	92	124	|	10101111000	111143
            93	GS	}	93	125	}	10100011110	111341
            94	RS	~	94	126	~	10001011110	131141
            95	US	DEL	95	-	-	10111101000	114113
            96	FNC3	FNC3	96	-	-	10111100010	114311
            97	FNC2	FNC2	97	-	-	11110101000	411113
            98	ShiftB	ShiftA	98	-	-	11110100010	411311
            99	CodeC	CodeC	99	-	-	10111011110	113141
            100	CodeB	FNC4	CodeB	-	-	10111101110	114131
            101	FNC4	CodeA	CodeA	-	-	11101011110	311141
            102	FNC1	FNC1	FNC1	-	-	11110101110	411131
            103	Start Start Start	208	SCA	11010000100	211412
            104	Start Start Start	209	SCB	11010010000	211214
            105	Start Start Start	210	SCC	11010011100	211232
            106	Stop Stop Stop	- -	11000111010	233111""".split()
    EAN13_CHART = """
            0   0001101 0100111 1110010
            1	0011001	0110011	1100110
            2	0010011	0011011	1101100
            3	0111101	0100001	1000010
            4	0100011	0011101	1011100
            5	0110001	0111001	1001110
            6	0101111	0000101	1010000
            7	0111011	0010001	1000100
            8	0110111	0001001	1001000
            9	0001011	0010111	1110100""".split()

    EAN13_MAP = {
            '111111': '0',
            '112122': '1',
            '112212': '2',
            '112221': '3',
            '121122': '4',
            '122112': '5',
            '122211': '6',
            '121212': '7',
            '121221': '8',
            '122121': '9'}
    if type == 'code_128':
        return CODE128_CHART
    if type == 'ean13':
        return EAN13_CHART, EAN13_MAP


def get_decoded_128_code(bits):
    sym_len = 11
    symbols = [bits[i:i + sym_len] for i in range(0, len(bits), sym_len)] #Разделение последовательности по 11 бит
    if symbols[0] == '11010000100':  #Проверка на тип кодировки
        code128 = CODE128A
    elif symbols[0] == '11010010000':
        code128 = CODE128B
    elif symbols[0] == '11010011100':
        code128 = CODE128C
    else:
        return None
    str_out = ""
    for sym in range(len(symbols)):          # Расшифровка по выбранному словарю
        if symbols[sym] and symbols[sym+1] in code128:
            if code128[symbols[sym]] == 'Start':
                continue
            if code128[symbols[sym]] == 'CodeA': # Смена словаря при поступлении команды
                code128 = CODE128A
                continue
            if code128[symbols[sym]] == 'CodeB':
                code128 = CODE128B
                continue
            if code128[symbols[sym]] == 'CodeC':
                code128 = CODE128C
                continue
            if code128[symbols[sym+1]] == 'Stop': # При поступлении команды стоп, прекратить расшифровку
                break
            str_out += code128[symbols[sym]]
        else:
            return None
    return str_out


def get_decoded_ean13(bits):
    sym_len = 7
    str_out = ""
    map = ""
    ean13_chart, ean13_map = encode_characters('ean13')
    symbols_1 = [value for value in ean13_chart[1::4]]
    symbols_2 = [value for value in ean13_chart[2::4]]
    symbols_3 = [value for value in ean13_chart[3::4]]
    values = [value for value in ean13_chart[0::4]]
    symbols_1 = dict(zip(symbols_1,values))
    symbols_2 = dict(zip(symbols_2, values))
    symbols_3 = dict(zip(symbols_3, values))
    left = bits[3:45]
    right = bits[50:92]
    l_symbols = [left[i:i + sym_len] for i in range(0, len(left), sym_len)]
    r_symbols = [right[i:i + sym_len] for i in range(0, len(right), sym_len)]
    symbols = l_symbols + r_symbols
    for sym in symbols:
        if sym in symbols_1:
            str_out += symbols_1[sym]
            map += '1'
        elif sym in symbols_2:
            str_out += symbols_2[sym]
            map += '2'
        elif sym in symbols_3:
            str_out += symbols_3[sym]
        else:
            return None
    if map in ean13_map:
        f_symbol = ean13_map[map]
    else:
        return None
    str_out = f_symbol + str_out
    return str_out


def decode(image):
    hor_data = collect_barcode_data(image)
    bit_width_code128, bit_width_ean13, pos1 = find_type_and_bit_width(hor_data)
    if isinstance(bit_width_ean13, int):
        # print(bit_width_ean13, basewidth, avg)
        bits_ean13 = get_code(hor_data, bit_width_ean13, pos1)
        answer = get_decoded_ean13(bits_ean13)
        print(bits_ean13)
        if answer:
            answer = [answer, "EAN_13"]
            return answer

    if isinstance(bit_width_code128, int):
        bits_code128 = get_code(hor_data, bit_width_code128, pos1)
        answer = get_decoded_128_code(bits_code128)
        if answer:
            answer = [answer, "CODE_128"]
            return answer
    return None


def run(image):
    answer = decode(image)
    if answer is None:
        image_180 = cv2.rotate(image, cv2.cv2.ROTATE_180)
        answer = decode(image_180)
    return answer


ENCODE = encode_characters('code_128')
SYMBOLS = [value for value in ENCODE[6::8]]
VALUESA = [value for value in ENCODE[1::8]]
VALUESB = [value for value in ENCODE[2::8]]
VALUESC = [value for value in ENCODE[3::8]]
CODE128A = dict(zip(SYMBOLS, VALUESA))
CODE128B = dict(zip(SYMBOLS, VALUESB))
CODE128C = dict(zip(SYMBOLS, VALUESC))


