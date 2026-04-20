#!/usr/bin/env python3

from decimal import Decimal

def decimal_range(start, stop, step):
    
    start = Decimal(str(start))
    stop = Decimal(str(stop))
    step = Decimal(str(step))
    
    current = start
    while current < stop:
        yield current
        current += step
