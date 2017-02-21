import ordinality_utilities as ord

import numpy as np
import math as m
import pandas as pd
import re as re
import io
import math as m

from IPython.display import display, HTML

#regex for phone numbers
'''supports
"1-234-567-8901",
"1-234-567-8901 x1234",
"1-234-567-8901 ext1234",
"1 (234) 567-8901",
"1.234.567.8901",
"1/234/567/8901",
"12345678901",'''
REGEX_PHONE = "1?\W*([0-9][0-9][0-9])\W*([2-9][0-9]{2})\W*([0-9]{4})(\se?x?t?(\d*))?"

#regex for email address
REGEX_EMAIL = "^([a-zA-Z0-9_\-\.]+)@([a-zA-Z0-9_\-\.]+)\.([a-zA-Z]{2,5})$"

#regex for measurements - distance, length
MEASURMENTS = {'', 'inch','inches','cm','miles','mil','mi',"'",'"','foot','feet','ft'}

#number regex comma separated
REGEX_NUMBER = ['^\d+$','^\d{1,3}(,\d{2,3})*(\.\d+)?$']

#regex for dates
REGEX_DATE = ['[\d]{1,2}(-|/)[\d]{1,2}(-|/)([\d]{4}|[\d]{2})', '[[\d]{1,2}(-|/)]{0,1}[\w]{1,3}(-|/)([\d]{4}|[\d]{2})',
             '[\d]{1,2}(-|/)([\d]{4}|[\d]{2})', '[\w]{1,3}(-|/)([\d]{4}|[\d]{2})','[\w]{1,3}(-|/)[\d]{1,2}(-|/)([\d]{4}|[\d]{2})',
             '[\d]{4}(-|/)[\d]{1,2}(-|/)([\d]{1,2})']

obj_type = 'object'

#generic function to test against a given regex
def _check_against_regex(regex, inp) :
    type_regex_variable = type(regex)

    #the regex should be list set or string
    if(type_regex_variable not in [list, set, str]) :
        print "Regex variable should be list, set or string"
        return

    #if the regex is of type list/set then see if it matches any of them
    #if the tye is a string check against it
    if(type_regex_variable == str) :
        regex = [regex]

    matched = [True if(re.match(reg, inp)) else False for reg in regex]
    return any(matched)

#check if the input is a phone number
'''supports
"1-234-567-8901",
"1-234-567-8901 x1234",
"1-234-567-8901 ext1234",
"1 (234) 567-8901",
"1.234.567.8901",
"1/234/567/8901",
"12345678901",'''
def is_phone_number(inp) :
    return _check_against_regex(REGEX_PHONE, inp) if (type(inp) == str) else False

#check if the input is email address
def is_email_address(inp) :
    return _check_against_regex(REGEX_EMAIL, inp) if (type(inp) == str) else False

#is a measurements - distance, length
def is_length(inp) :
    if (type(inp) != str) :
        return False
    inp = re.sub("\\s{1,}","",inp)
    return (len(set(re.split("[\d]+", inp,flags=re.IGNORECASE)) - MEASURMENTS) == 0)

# geo location

# numbers, comma separated numbers
def is_number(inp) :
    if(type(inp) == str) :
        return _check_against_regex(REGEX_NUMBER, inp)
    else :
        try :
            float(inp)
            return True
        except :
            return False

# numbers, comma separated numbers
def is_date(inp) :
    return _check_against_regex(REGEX_DATE, inp) if (type(inp) == str) else False

#TODO date and time

#map type to function
TYPE_FUNCTION = {'NUMBER': is_number,
                'DATE':is_date,
                'DISTANCE':is_length,
                'PHONE':is_phone_number,
                'EMAIL':is_email_address}

#return list of all the possible types
def get_types(inp) :
    if(not inp) | (inp == np.NaN) | (inp == None):
        return []
    if(type(inp) == float) :
        if(m.isnan(inp) == True) :
            return []
    return [key for key in TYPE_FUNCTION if TYPE_FUNCTION[key](inp) == True]


'''
input : pandas column - series
'''
def get_column_types(unique_values) :
    other_types = []
    types = {}

    unique_values = pd.unique(unique_values)

    return {y for x in unique_values for y in get_types(x) }

def info(df) :
    buf = io.StringIO()
    d = df.info(buf=buf)
    info = buf.getvalue()
    lines = info.split("\n")
    col_row_start = -1
    tot_rows = -1
    for i in xrange(len(lines)) :
        if(lines[i].startswith('Data')) :
            break
    items = []
    for line in lines[i+1:] :
        if(line.startswith(u'dtypes')) :
            break
        loc = line.split()
        un = pd.unique(df[loc[0]])
        if(loc[3] == obj_type) :
            types = get_column_types(un)
            val = types if len(get_column_types(un)) > 0 else None
            if not val:
                types = ord.identify_ordinal_type(un)
                val = types[1] + " ordinal?" if types[0] else ""
            loc.append(val if val else "")
        else :
            loc.append("")
        loc.append(un)
        items.append(loc)

    df3 = pd.DataFrame(items,columns=['Column','Count','','Type','Possible Values','Few Values'])
    df3.index = df3['Column']
    del df3['Column']
    print display(df3)
