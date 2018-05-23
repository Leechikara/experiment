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

QA_PERMITTED_ATTR = ['price', 'discountValue', 'weight', 'os', 'color', 'thickness', 'size', 'material',
                     'network', 'nfc', 'generation']
COMPARE_PERMITTED_ATTR = ['price', 'discountValue', 'weight', 'thickness', 'size', 'material',
                          'network', 'generation']
CONFIRM_PERMITTED_ATTR = ['price', 'discountValue', 'weight', 'os', 'color', 'thickness', 'size', 'material',
                          'network', 'nfc', 'generation']
PRE_SALES_ATTR = list(set(QA_PERMITTED_ATTR) | set(COMPARE_PERMITTED_ATTR) | set(CONFIRM_PERMITTED_ATTR))

OTHER_ATTRIBUTE_DEFINITION = {
    'payment': {'dtype': 'constantList', 'value': ['weChat', 'Alipay', 'creditCard', 'debitCard']},
    'expressTime': {'dtype': 'constantInt', 'value': 3},
    'expressName': {'dtype': 'constantList', 'value': ['顺丰', '韵达', '菜鸟', '中通']}}

# we simulate the complicity of dialog
AVAILABLE_INTENT_1 = {'pre_sales': ['qa'],
                      'in_sales': ['payment', 'discountURL', 'gift', 'expressTime', 'expressName']}

AVAILABLE_INTENT_2 = {'pre_sales': ['qa', 'confirm'],
                      'in_sales': ['payment', 'discountURL', 'gift', 'expressTime', 'expressName']}

AVAILABLE_INTENT_3 = {'pre_sales': ['qa', 'confirm', 'compare'],
                      'in_sales': ['payment', 'discountURL', 'gift', 'expressTime', 'expressName']}

AVAILABLE_INTENT_4 = {'pre_sales': ['qa', 'confirm', 'compare'],
                      'in_sales': ['payment', 'discountURL', 'gift', 'expressTime', 'expressName'],
                      'after_sales': ['exchange']}

AVAILABLE_INTENT_5 = {'pre_sales': ['qa', 'confirm', 'compare'],
                      'in_sales': ['payment', 'discountURL', 'gift', 'expressTime', 'expressName'],
                      'after_sales': ['exchange', 'consult', 'refund', 'expressInfo', 'gift', 'invoice']}

AVAILABLE_INTENT_6 = {'pre_sales': ['qa', 'confirm', 'compare'],
                      'in_sales': ['payment', 'discountURL', 'gift', 'expressTime', 'expressName'],
                      'after_sales': ['consult', 'refund', 'expressInfo', 'gift', 'invoice', 'exchange'],
                      'sentiment': ['positive', 'negative']}

# intent and grammar sample probability in different scenes
INTENT_P_DICT = {'pre_sales': {'qa': 1 / 3, 'confirm': 1 / 3, 'compare': 1 / 3}}
GRAMMAR_P_DICT = {'pre_sales': {'qa': {'init': {'complete': 1 / 30, 'lack_attribute': 1 / 3, 'lack_entity': 1 / 3},
                                       'horizontal_1': {'complete': 1 / 20, 'lack_attribute': 1 / 2},
                                       'vertical_1': {'complete': 1 / 20, 'lack_entity': 1 / 2},
                                       'diagonal_1': {'complete': 1 / 10}},
                                'confirm': {'init': {'complete': 1 / 20, 'lack_entity': 1 / 2},
                                            'horizontal_1': {'complete': 1 / 10},
                                            'vertical_1': {'complete': 1 / 20, 'lack_entity': 1 / 2},
                                            'diagonal_1': {'complete': 1 / 10}},
                                'compare': {'init': {'complete': 1 / 40, 'lack_entity': 1 / 4,
                                                     'lack_attribute': 1 / 4, 'lack_attribute_entity': 1 / 4},
                                            'horizontal_1': {'complete': 1 / 40, 'lack_entity': 1 / 4,
                                                             'lack_attribute': 1 / 4, 'lack_attribute_entity': 1 / 4},
                                            'diagonal_1': {'complete': 1 / 20},
                                            'horizontal_2': {'complete': 1 / 20, 'lack_attribute': 1 / 2},
                                            'diagonal_2': {'complete': 1 / 10}}}}
