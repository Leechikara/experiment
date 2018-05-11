# coding=utf-8

"""
Generate reasonable episodes in customer service.

There are five principles:
1. All episodes are based on basic scenes defined in scrip.txt
2. The context is important. Some turns can be resolved according to context.
3. Defining reasonable episodes is non-trivial. We must verify the rationalisation.
4. Dialog flows based on entity, attribute, intent.
5. Not all basic scenes are usable. We hold a switch to control that.

We think such episodes are reasonable:
    pre_sales, in_sales, after_sales, pre_sales + in_sales, sentiment + all combinations aforementioned

    In pre_sales:

    In in_sales:

    In after_sales:

    In pre_sales + in_sales:

    In sentiment + all combinations aforementioned:


"""
