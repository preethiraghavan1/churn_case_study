
ordinal_dic = {'small_medium_large':["mini", 'extra small' 'small', "compact", "medium", 'large / medium', 'large', 'extra large']
               ,'s_m_l' : ['xs','s','m','l','xl','xxl','xxxl']
,'low_medium_high' : ['low','medium','high']
,'l_m_h' : ['l','m','h']
,'yes_no' : ['no','yes']
,'y_n' : ['n','y']
,'true_false' : [False,True]
}

def contains_ordinal(unique_values, existing_ordinas) :
    return len(set(unique_values) - set(existing_ordinas)) < len(set(unique_values))

def identify_ordinal_type(unique_values) :
#       if(not unique_values) :
#             return None
      unique_values = [x.lower() if type(x) == str else x for x in unique_values]
      ret_dic = None
      for ord_type in ordinal_dic :
            #if the unique values are in pre defined dictionary
            if(contains_ordinal(unique_values, ordinal_dic[ord_type])) :
                # form a dictionary with the correct order
                # example : if input = ['small','large']
                #     return {'small':1, 'large':2}
                #if input = ['small', 'medium', 'large']
                #     return {'small':1, 'medium':2,'large':3}
                ret_dic = {v:key+1 for key,v in enumerate(sorted(unique_values, key=lambda x : ordinal_dic[ord_type].index(x) if x in ordinal_dic[ord_type] else -1))}
                break
#       print str(unique_values), "is", "" if ret_dic else "not", "a known ordinal"
      return (ret_dic, ord_type)
