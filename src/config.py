# coding=utf-8
DATA_ROOT = '/home/wkwang/workstation/experiment/data'

ORIGINAL_ENTITY = 50
CHANGE_ORIGINAL_ENTITY = False
ADD_NEW_ENTITY = 0

GOODS_ATTRIBUTE_DEFINITION = {'price': {'dtype': 'int', 'min': 1000, 'max': 2000},
                              'discountValue': {'dtype': 'float', 'range': [0.1 * x for x in range(6, 10)],
                                                'acceptNone': True},
                              'discountURL': {'dtype': 'str', 'prefix': 'https://activity/', 'entity_id': None,
                                              'discountValue': None, 'postfix': '.html', 'correlate': 'discountValue'},
                              'weight': {'dtype': 'int', 'range': range(80, 100), 'expensive': 'low'},
                              'os': {'dtype': 'str', 'range': ['Android', 'iOS']},
                              'color': {'dtype': 'list', 'range': ['blue', 'black', 'yellow', 'red']},
                              'thickness': {'dtype': 'float', 'range': [0.1 * x for x in range(30, 41)],
                                            'expensive': 'low'},
                              'size': {'dtype': 'float', 'range': [0.1 * x for x in range(90, 100)],
                                       'expensive': 'high'},
                              'material': {'dtype': 'str', 'range': ['塑料', '木', '金属'], 'expensive': 'high'},
                              'network': {'dtype': 'str', 'range': ['3g', '4g'], 'expensive': 'high'},
                              'nfc': {'dtype': 'bool', 'range': [True, False], 'expensive': 'low'},
                              'generation': {'dtype': 'str', 'range': [str(x) for x in range(1, 4)],
                                             'expensive': 'high'},
                              'gift': {'dtype': 'str', 'range': ['表带', '耳机', '电池']}}
PRICE_NOISE = 400

OTHER_ATTRIBUTE_DEFINITION = {'payment': {}, 'expressTime': {}, 'expressName': {}}
