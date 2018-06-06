# coding=utf-8


DATA_ROOT = "/home/wkwang/workstation/experiment/data"

# experiments for KB changes
ORIGINAL_ENTITY = 50
CHANGE_ORIGINAL_ENTITY = False
NEW_ADD_ENTITY = 0

# attribute definition of goods
GOODS_ATTRIBUTE_DEFINITION = {"price": {"dtype": "int", "min": 1000, "max": 2000, "prefer": "low", "unit": "RMB"},
                              "discountValue": {"dtype": "int", "range": [x for x in range(6, 10)],
                                                "acceptNone": True, "prefer": "low", "unit": "折"},
                              "discountURL": {"dtype": "str", "prefix": "https://activity/", "entity_id": None,
                                              "discountValue": None, "postfix": ".html", "correlate": "discountValue"},
                              "weight": {"dtype": "int", "range": range(80, 100), "expensive": "low",
                                         "prefer": "low", "unit": "克"},
                              "os": {"dtype": "str", "range": ["安卓", "苹果"]},
                              "color": {"dtype": "list", "range": ["蓝色", "黑色", "黄色", "红色", "白色", "绿色"]},
                              "thickness": {"dtype": "float", "range": [0.1 * x for x in range(5, 21)],
                                            "expensive": "low", "prefer": "low", "unit": "厘米"},
                              "size": {"dtype": "float", "range": [0.1 * x for x in range(90, 100)],
                                       "expensive": "high", "prefer": "high", "unit": "平方厘米"},
                              "material": {"dtype": "str", "range": ["塑料", "木头", "金属"], "expensive": "high",
                                           "prefer": "high"},
                              "network": {"dtype": "str", "range": ["3g", "4g"], "expensive": "high",
                                          "prefer": "high"},
                              "nfc": {"dtype": "bool", "range": [True, False], "expensive": "low",
                                      "prefer": "low"},
                              "generation": {"dtype": "str", "range": [str(x) for x in range(1, 4)],
                                             "expensive": "high", "prefer": "high", "unit": "代"}}

# global attribute definitions
OTHER_ATTRIBUTE_DEFINITION = {
    "payment": {"dtype": "constantList", "value": ["微信", "支付宝", "信用卡"]},
    "expressTime": {"dtype": "constantInt", "value": 3},
    "expressName": {"dtype": "constantStr", "value": "顺丰"},
    "osUpdate": {"dtype": "constantStr", "value": "https://osUpdate.html"},
    "networkReset": {"dtype": "constantStr", "value": "https://networkRest.html"},
    "nfcReset": {"dtype": "constantStr", "value": "https://nfcReset.html"}
}

# simulate noise in price
PRICE_NOISE = 300

# available attributes in different intents
QA_PERMITTED_ATTR = ["price", "discountValue", "weight", "os", "color", "thickness", "size", "material",
                     "network", "nfc", "generation"]
COMPARE_PERMITTED_ATTR = ["price", "discountValue", "weight", "thickness", "size", "material",
                          "network", "generation"]
CONFIRM_PERMITTED_ATTR = ["price", "discountValue", "weight", "os", "color", "thickness", "size", "material",
                          "network", "nfc", "generation"]
PRE_SALES_ATTR = ["price", "discountValue", "weight", "os", "color", "thickness", "size", "material",
                  "network", "nfc", "generation"]

# we simulate the size of Hypothesis Space
AVAILABLE_INTENT_1 = {"pre_sales": ["qa"],
                      "in_sales": ["payment", "discountURL", "expressTime", "expressName"]}

AVAILABLE_INTENT_2 = {"pre_sales": ["qa", "confirm"],
                      "in_sales": ["payment", "discountURL", "expressTime", "expressName"]}

AVAILABLE_INTENT_3 = {"pre_sales": ["qa", "confirm", "compare"],
                      "in_sales": ["payment", "discountURL", "expressTime", "expressName"]}

AVAILABLE_INTENT_4 = {"pre_sales": ["qa", "confirm", "compare"],
                      "in_sales": ["payment", "discountURL", "expressTime", "expressName"],
                      "after_sales": ["expressInfo", "invoice", "exchange", "exchange_exchange", "consult", "refund",
                                      "consult_refund"]}

AVAILABLE_INTENT_5 = {"pre_sales": ["qa", "confirm", "compare"],
                      "in_sales": ["payment", "discountURL", "expressTime", "expressName"],
                      "after_sales": ["expressInfo", "invoice", "exchange", "exchange_exchange", "consult", "refund",
                                      "consult_refund"],
                      "sentiment": ["positive", "negative"]}

# intent distributions in different scenes
INTENT_P_DICT = {"pre_sales": {"qa": 1 / 3, "confirm": 1 / 3, "compare": 1 / 3},
                 "in_sales": {"discountURL": 1 / 4, "payment": 1 / 4, "expressTime": 1 / 4,
                              "expressName": 1 / 4},
                 "after_sales": {"consult": 1 / 7, "refund": 1 / 7, "exchange": 1 / 7, "expressInfo": 1 / 7,
                                 "invoice": 1 / 7, "consult_refund": 1 / 7, "exchange_exchange": 1 / 7}}

# grammar distribution in different situations
# We can change this distributes to improve the quality of episode generation rules!
GRAMMAR_P_DICT = {"pre_sales": {"qa": {"init": {"complete": 1 / 3, "lack_attribute": 1 / 3, "lack_entity": 1 / 3},
                                       "horizontal_1": {"complete": 1 / 2, "lack_attribute": 1 / 2},
                                       "vertical_1": {"complete": 1 / 2, "lack_entity": 1 / 2},
                                       "diagonal_1": {"complete": 1}},
                                "confirm": {"init": {"complete": 1 / 2, "lack_entity": 1 / 2},
                                            "horizontal_1": {"complete": 1},
                                            "vertical_1": {"complete": 1 / 2, "lack_entity": 1 / 2},
                                            "diagonal_1": {"complete": 1}},
                                "compare": {"init": {"complete": 1 / 4, "lack_entity": 1 / 4,
                                                     "lack_attribute": 1 / 4, "lack_attribute_entity": 1 / 4},
                                            "horizontal_2": {"complete": 1 / 2, "lack_attribute": 1 / 2},
                                            "diagonal_2": {"complete": 1}}},
                  "in_sales": {"discountURL": {"complete": 1 / 2, "lack_entity": 1 / 2},
                               "payment": {"complete": 1},
                               "expressTime": {"complete": 1},
                               "expressName": {"complete": 1}}}

# all possible sentiment target. Define first, but some are not existing, for example "generation".
SENTIMENT_TARGET = ["general", "price", "color", "size", "discountValue", "generation",
                    "discountURL", "material", "weight", "thickness", "pre_sales_end",
                    "expressTime", "expressInfo", "network", "nfc", "os", "payment", "expressName"]
SENTIMENT_RULES = ["append", "prefix", "insert", "mix"]

# rule for reorganize exchange&exchange
REORGANIZE_RULES = ["None", "insert", "mix"]
