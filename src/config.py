# coding=utf-8
DATA_ROOT = '/home/wkwang/workstation/experiment/data'
MAX_ENTITY = 50
NEW_ENTITY = 0

GOODS_ATTRIBUTE_DEFINITION = [{'index': 0, 'key': 'price', 'dtype': 'int', 'min': 1000, 'max': 2000},
                              {'index': 1, 'key': 'discountValue', 'dtype': 'float',
                               'range': [0.1 * x for x in range(6, 10)],
                               'acceptNone': True},
                              {'index': 2, 'key': 'discountURL', 'dtype': 'str', 'prefix': 'https://activity/',
                               'entity_id': None, 'discountValue': None,
                               'postfix': '.html', 'correlate': 'discountValue'},
                              {'index': 3, 'key': 'weight', 'dtype': 'float', 'range': range(80, 100),
                               'expensive': 'Low'},
                              {'index': 4, 'key': 'os', 'dtype': 'str', 'range': ['Android', 'iOS']},
                              {'index': 5, 'key': 'color', 'dtype': 'list',
                               'range': ['blue', 'black', 'yellow', 'red']},
                              {'index': 6, 'key': 'thickness', 'dtype': 'float',
                               'range': [0.1 * x for x in range(30, 41)],
                               'expensive': 'Low'},
                              {'index': 7, 'key': 'size', 'dtype': 'float', 'range': [0.1 * x for x in range(90, 100)]},
                              {'index': 8, 'key': 'material', 'dtype': 'str', 'range': ['塑料', '木', '金属'],
                               'expensive': 'high'},
                              {'index': 9, 'key': 'network', 'dtype': 'str', 'range': ['3g', '4g'],
                               'expensive': 'high'},
                              {'index': 10, 'key': 'nfc', 'dtype': 'bool', 'range': [True, False], 'expensive': 'low'},
                              {'index': 11, 'key': 'generation', 'dtype': 'int', 'range': range(1, 4),
                               'expensive': 'high'},
                              {'index': 12, 'key': 'gift', 'dtype': 'str', 'range': ['表带', '耳机', '电池']}]
PRICE_NOISE = 400

OTHER_ATTRIBUTE_DEFINITION = ['payment', 'expressTime', 'expressName']
