# coding=utf-8
DATA_ROOT = '/home/wkwang/workstation/experiment/data'

ORIGINAL_ENTITY = 50
CHANGE_ORIGINAL_ENTITY = False
NEW_ADD_ENTITY = 0

GOODS_ATTRIBUTE_DEFINITION = {'price': {'dtype': 'int', 'min': 1000, 'max': 2000, 'prefer': 'low'},
                              'discountValue': {'dtype': 'float', 'range': [0.1 * x for x in range(6, 10)],
                                                'acceptNone': True, 'prefer': 'low'},
                              'discountURL': {'dtype': 'str', 'prefix': 'https://activity/', 'entity_id': None,
                                              'discountValue': None, 'postfix': '.html', 'correlate': 'discountValue'},
                              'weight': {'dtype': 'int', 'range': range(80, 100), 'expensive': 'low',
                                         'prefer': 'low'},
                              'os': {'dtype': 'str', 'range': ['Android', 'iOS']},
                              'color': {'dtype': 'list', 'range': ['blue', 'black', 'yellow', 'red']},
                              'thickness': {'dtype': 'float', 'range': [0.1 * x for x in range(30, 41)],
                                            'expensive': 'low', 'prefer': 'low'},
                              'size': {'dtype': 'float', 'range': [0.1 * x for x in range(90, 100)],
                                       'expensive': 'high', 'prefer': 'high'},
                              'material': {'dtype': 'str', 'range': ['塑料', '木', '金属'], 'expensive': 'high',
                                           'prefer': 'high'},
                              'network': {'dtype': 'str', 'range': ['3g', '4g'], 'expensive': 'high',
                                          'prefer': 'high'},
                              'nfc': {'dtype': 'bool', 'range': [True, False], 'expensive': 'low',
                                      'prefer': 'low'},
                              'generation': {'dtype': 'str', 'range': [str(x) for x in range(1, 4)],
                                             'expensive': 'high', 'prefer': 'high'},
                              'gift': {'dtype': 'str', 'range': ['表带', '耳机', '电池']}}
PRICE_NOISE = 300

OTHER_ATTRIBUTE_DEFINITION = {
    'payment': {'dtype': 'constantList', 'value': ['weChat', 'Alipay', 'creditCard', 'debitCard']},
    'expressTime': {'dtype': 'constantInt', 'value': 3},
    'expressName': {'dtype': 'constantList', 'value': ['顺丰', '韵达', '菜鸟', '中通']}}
