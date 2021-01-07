import numpy as np
import os
import pickle
import psycopg2 as pg
import sys
import warnings
import csv
import warnings
import functools
import Levenshtein as lev

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used.

    from https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


def getfilenames(directory='./', extension=None):
    names = []
    if extension is None:
        return os.listdir(directory)

    for file in os.listdir(directory):
        if file.endswith(extension):
            names.append(directory + "/" + file)
    return names


def read_paired_data_file(filename, delim='=', ignore='--'):
    data = dict()

    with open(filename, 'r', errors='replace') as f:
        info = np.array(f.readlines())

        if ignore is not None:
            info = [i.strip() for i in
                    info[np.argwhere([j not in ['\n'] and not j.startswith(ignore) for j in info]).reshape((-1))]]

        for i in info:
            pair = [j.strip() for j in i.split(delim)]
            if len(pair) != 2:
                print('\n\033[91m', 'ERROR - Unable to parse the line: {}'.format(i), '\033[0m\n')
                sys.stdout.flush()
            else:
                data[pair[0].lower()] = pair[1]

    return data


def read_text_file(filename, sep=None, ignore='--'):
    text = '' if sep is not None else []

    with open(filename, 'r', errors='replace') as f:
        info = np.array(f.readlines())

        info[0] = str(info[0]).replace('ï»¿', '')

        if ignore is not None:
            info = [i.strip() for i in
                    info[np.argwhere([j not in ['\n'] and not j.startswith(ignore) for j in info]).reshape((-1))]]

        for i in info:
            if sep is not None:
                text += i + sep
            else:
                text.append(i)

    return text


def __load_csv__(filename, max_rows=None):
    csvarr = []
    n_lines = len(open(filename).readlines())
    with open(filename, 'r', errors='replace') as f:
        f_lines = csv.reader(f)

        # n_lines = sum(1 for row in f_lines)
        # print(n_lines)
        # n_lines = sum(1 for row in f_lines)
        if max_rows is not None:
            n_lines = max_rows

        output_str = '-- loading {}...({}%)'.format(filename, 0)
        sys.stdout.write(output_str)
        sys.stdout.flush()
        old_str = output_str
        i = 0
        for line in f_lines:
            if len(line) == 0:
                continue
            line = np.array(line)
            na = np.argwhere(np.array(line[:]) == '#N/A').ravel()
            if len(na) > 0:
                line[na] = ''

            na = np.argwhere(np.array(line[:]) == 'NA').ravel()

            if len(na) > 0:
                line[na] = ''

            csvarr.append(line)
            if max_rows is not None:
                if len(csvarr) >= max_rows:
                    break
            if not round((i / n_lines) * 100, 2) == round(((i - 1) / n_lines) * 100, 2):
                sys.stdout.write('\r' + (' ' * len(old_str)))
                output_str = '\r-- loading {}...({}%)'.format(filename, round((i / n_lines) * 100, 2))
                sys.stdout.write(output_str)
                sys.stdout.flush()
                old_str = output_str

            i += 1
        sys.stdout.write('\r' + (' ' * len(old_str)))
        sys.stdout.write('\r-- loading {}...({}%)\n'.format(filename, 100))
        sys.stdout.flush()

    return csvarr


def write_csv(data, filename, headers=None, append=False):

    if headers is None:
        headers = []

    if not filename.endswith('.csv'):
        filename += '.csv'

    with open(filename, 'w' if not append else 'a') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')

        if len(headers)!=0:
            writer.writerow(np.array(headers, dtype=str))
            # for i in range(0,len(headers)-1):
            #     f.write(str(headers[i]) + ',')
            # f.write(str(headers[len(headers)-1])+'\n')
        # for i in range(0,len(data)):
        # ar = np.array(data, dtype=str)
        # ar = ar.reshape((ar.shape[0],-1))

        for j in data:
            row = np.array(j, dtype=str)
            row[np.argwhere([k == 'None' for k in row]).ravel()] = ''
            row[np.argwhere([k is None for k in row]).ravel()] = ''
            writer.writerow(row)

    f.close()


def read_csv(filename, max_rows=None, headers=True):
    if max_rows is not None:
        max_rows += 1

    if not filename.endswith('.csv'):
        filename += '.csv'

    data = __load_csv__(filename,max_rows)

    if headers:
        headers = np.array(data[0])
        data = np.delete(data, 0, 0)
        return data, headers
    else:
        return data


def read_csv_headers(filename):
    if not filename.endswith('.csv'):
        filename += '.csv'

    with open(filename, 'r') as f:
        for line in f.readlines():
            return line.strip().split(',')
    return []


def pickle_save(instance, filename):
    pickle.dump(instance, open(filename, "wb"), -1)


def pickle_load(filename):
    return pickle.load(open(filename, "rb"))


def infer_if_string(ar, n=None):
    ar = np.array(ar)
    assert len(ar.shape) == 1

    if n is None:
        n = ar.shape[0]
    else:
        n = np.minimum(ar.shape[0],n)

    for i in range(n):
        try:
            float(ar[i])
        except ValueError:
            if ar[i] == '':
                continue
            else:
                return True
    return False


def infer_basic_type(ar, n=None):
    ar = np.array(ar)
    assert len(ar.shape) == 1

    if n is None:
        n = ar.shape[0]
    else:
        n = np.minimum(ar.shape[0],n)

    is_int = True

    for i in range(n):
        try:
            temp = float(ar[i])
            if not temp == int(temp):
                is_int = False
        except ValueError:
            if ar[i] == '':
                continue
            else:
                return 'text'
    return 'double precision' if not is_int else 'integer'


def as_factor(ar, return_labels=False):
    ar = np.array(ar).reshape((-1))
    label = np.unique(ar)
    for i in range(len(label)):
        ar[ar[:] == label[i]] = i
    if return_labels:
        lab = [label[int(i)] for i in ar]
        return ar, lab
    return ar


def as_float(ar):
    ar = np.array(ar, dtype=str)
    ar[ar == '.'] = float('nan')
    ar[ar == ' '] = float('nan')
    ar[ar == ''] = float('nan')
    return np.array(ar, dtype=np.float32).reshape((-1))


def nan_omit(ar):
    ar = np.array(ar, dtype=str).reshape((-1))
    if not infer_if_string(ar):
        ar = ar[np.where(ar[:] != '')]
        ar = np.array(ar, dtype=np.float32)
        ar = ar[np.where(ar[:] != float('nan'))]
    else:
        ar = ar[np.where(ar[:] != '')]
    return ar


def one_hot(ar, class_array, class_column, replace=False):
    npar = np.array(ar)

    classes = class_array

    enc = np.zeros(shape=(len(ar), len(classes)), dtype=np.float32)
    str('asdf').lower()

    for i in range(len(npar)):
        enc[i,
            np.argwhere(np.array([str(c).lower() for c in classes]) == str(npar[i, class_column]).lower()).ravel()] = 1
    for i in range(len(classes)):
        npar = np.insert(npar, len(npar[0,:]), values=enc[:, i], axis=1)

    if replace:
        return np.delete(npar,class_column,axis=1)

    return npar


def cross_feature(class_array, feature_array, fill=0, distinct_classes=None, distinct_feature_values=None):
    assert len(class_array) == len(feature_array)

    ca = np.array(class_array).reshape((-1))
    fa = np.array(feature_array).reshape((len(feature_array), -1))

    if not hasattr(distinct_feature_values[0],'__iter__'):
        distinct_feature_values = [distinct_feature_values]

    if distinct_feature_values is not None:
        assert len(distinct_feature_values) == fa.shape[1]

    c_feature = {'class_label': [], 'cross_feature': []}

    if fill is None:
        fill = np.nan

    if distinct_classes is None:
        dc = np.unique(class_array)
    else:
        dc = np.unique(np.array(distinct_classes).reshape((-1)))

    n_cross = 0

    value_lookup = []
    for i in range(fa.shape[1]):
        if distinct_feature_values is not None:
            dfv = np.unique(distinct_feature_values[i])
        else:
            dfv = np.unique(fa[:,i])

        value_lookup.append({'offset': n_cross, 'values': dfv})
        n_cross += len(dfv)

    for i in range(len(ca)):
        row = np.ones((n_cross*len(dc))) * fill

        ind = np.argwhere(dc == ca[i]).ravel()
        if len(ind) == 0:
            raise LookupError('A class label exists in the data that is not defined within the distinct class labels')
        ind = ind[0]

        for j in range(n_cross):
            row[(n_cross * ind) + j] = 0

        for j in range(fa.shape[1]):
            f_ind = np.argwhere(np.array(value_lookup[j]['values'], str) == str(fa[i,j])).ravel()
            if len(f_ind) == 0:
                raise LookupError(
                    'The value {} exists in the data but is not defined within the distinct feature values'
                        .format(str(fa[i,j])))
            f_ind = f_ind[0]

            row[(n_cross * ind) + value_lookup[j]['offset'] + f_ind] = 1

        c_feature['class_label'].append(ca[i])
        c_feature['cross_feature'].append(row)

    c_feature['class_label'] = np.array(c_feature['class_label'])
    c_feature['cross_feature'] = np.array(c_feature['cross_feature'])
    return c_feature


def levenshtein_ratio(ar, text_column, grouping=None, order=None, replace=False):
    table = np.array(ar)

    if grouping is None:
        grouping = table.shape[1]
        table = np.append(table, np.ones((table.shape[0], 1)), axis=1)

    if hasattr(grouping, '__iter__'):
        pivot_ind = table.shape[1]
        table = np.append(table, np.array(['~'.join(table[i, grouping]) for i in range(len(table))],
                                          dtype=str).reshape((-1, 1)), 1)
        pivot = pivot_ind
    else:
        pivot = grouping
    try:
        tbl_order = np.array(table[:, pivot], dtype=np.float32)
    except ValueError:
        tbl_order = np.array(table[:, pivot], dtype=str)

    _, piv = np.unique(tbl_order, return_index=True)
    p = table[piv, pivot]

    repl_col = text_column
    if not replace:
        repl_col = table.shape[1]
        table = np.append(table,np.zeros((len(table),1)),axis=1)

    for i in p:

        match = np.argwhere(np.array(table[:, pivot]) == i).ravel()
        filtered_table = table[match]

        if order is None:
            ordering = list(range(len(match)))
        else:
            if not hasattr(order, '__iter__'):
                order = [order]

            lex_order = []
            for j in range(len(order)):
                ord_j = len(order) - 1 - j
                lex_order.append(filtered_table[:, order[ord_j]])

            ordering = np.lexsort(tuple(lex_order))

        if not hasattr(text_column, '__iter__'):
            text_column = [text_column]

        ftable = filtered_table[ordering]
        lev_col = ftable.shape[1]
        filtered_table = np.append(filtered_table,np.zeros((len(filtered_table),1)),axis=1)
        for k in text_column:
            for j in range(1, len(ftable)):
                filtered_table[ordering[j],lev_col] = lev.ratio(str(filtered_table[ordering[j], k]),
                                                                str(filtered_table[ordering[j-1], k]))
        table[match, repl_col] = filtered_table[:,-1]

    return np.delete(table,pivot,axis=1)

    # classes = class_array
    #
    # enc = np.zeros(shape=(len(ar), len(classes)), dtype=np.float32)
    #
    # for i in range(len(npar)):
    #     enc[i, np.argwhere(np.array(classes[:], dtype=str) == str(npar[i, class_column])).ravel()] = 1
    # for i in range(len(classes)):
    #     npar = np.insert(npar, len(npar[0,:]), values=enc[:, i], axis=1)
    #
    # return npar


def print_descriptives(ar, headers=None, desc_level=1):
    ar = np.array(ar)
    ar = ar.reshape((-1,ar.shape[-1]))

    if headers is not None:
        # assert len(headers) == ar.shape[-1]
        headers = [str(i) + ' ' + headers[i] if i < len(headers) else 'Covariate ' + str(i) for i in
                   range(ar.shape[-1])]
    else:
        headers = ['Covariate ' + str(i) for i in range(ar.shape[-1])]

    print("{:=<{size}}".format('', size=50 + (30 * desc_level)))
    print("{:<15}{:^25}".format('DESCRIPTIVES', "{} Rows, {} Columns".format(ar.shape[0],ar.shape[1])))
    print("{:=<{size}}".format('', size=50 + (30 * desc_level)))
    for i in range(ar.shape[-1]):
        h = headers[i]
        if len(h) > 15:
            h = ''.join(list(h)[:15]) + '...'
        label = "Column {}".format(i) if headers is None else h
        dtype = ['int','float','string'][np.array(np.where(
            np.array(['integer','double precision','text'])[:] == infer_basic_type(np.unique(ar[:, i]), 1000))).reshape((-1))[0]]
        label = "{} ({}):".format(label,dtype)

        if dtype == 'string':
            m = np.array(np.array(ar[:, i]) == '').sum()
            desc1 = "{} unique values".format(len(np.unique(ar[:, i])))
            desc2 = ''
            desc3 = ''
        else:
            ar[:, i][ar[:, i] == ''] = float('nan')
            f_ar = np.array(ar[:, i], dtype=np.float32)
            m = np.isnan(f_ar).sum()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                desc1 = "mean={:<.2f} (SD={:<.2f})".format(np.nanmean(f_ar), np.nanstd(f_ar))
                desc2 = "median = {:<.2f}".format(np.nanmedian(f_ar))
                desc3 = "min={:<.2f}, max={:<.2f}".format(np.nanmin(f_ar), np.nanmax(f_ar))
        missing = "{} missing ({:<.1f}%)".format(m, m / float(ar.shape[0]))
        print("{:<30} {:<20} {:<35} {:<30} {:<30}".format(label, missing,
                                                          desc1 if desc_level > 0 else '',
                                                          desc2 if desc_level > 1 else '',
                                                          desc3 if desc_level > 2 else ''))
    print("{:=<{size}}\n".format('', size=50 + (30 * desc_level)))


def ndims(ar):
    d = 0
    a = ar

    while hasattr(a, '__iter__'):
        d += 1
        a = a[0]

        try:
            _ = '0' + a
            break
        except TypeError:
            pass

    return d


def db_connect(db_name, user, password='', host='127.0.0.1', port='5432'):
    # try:
    return pg.connect(dbname=db_name, user=user, password=password, host=host, port=port)
    # except Exception:
    #     return None


def db_query(db_object, query, arguments=None, return_column_names=False):
    assert type(arguments) is dict or arguments is None

    if arguments is not None:
        for k in arguments:
            if not isinstance(arguments[k],str) and hasattr(arguments[k], '__iter__'):



                query = query.replace(str(k), ','.join(arguments[k]))
            else:
                query = query.replace(str(k), '\'' + str(arguments[k]) + '\'' if isinstance(arguments[k], str) else str(
                    arguments[k]))

    cur = db_object.cursor()
    try:
        cur.execute(query)
    except Exception:
        import traceback
        print('\033[91m')
        traceback.print_exc(file=sys.stdout)
        print(query + '\033[0m')

    try:
        if return_column_names:
            return cur.fetchall(), [desc[0] for desc in cur.description]
        else:
            return cur.fetchall()
    except Exception:
        try:
            db_object.commit()
            return []
        except Exception:
            return None


def db_query_fetch(db_object, query, arguments=None, return_column_names=False):
    assert type(arguments) is dict or arguments is None

    if arguments is not None:
        for k in arguments:
            query = query.replace(str(k), '\'' + str(arguments[k]) + '\'' if isinstance(arguments[k], str) else str(
                arguments[k]))

    cur = db_object.cursor()
    try:
        cur.execute(query)
    except Exception:
        import traceback
        print('\033[91m')
        traceback.print_exc(file=sys.stdout)
        print(query + '\033[0m')

    if return_column_names:
        return cur.fetchall(), [desc[0] for desc in cur.description]
    else:
        return cur.fetchall()


def db_pull_data(csv_out_filename, db_object, query, arguments=None, partition=10000):
    # s_query = 'SELECT COUNT(*) FROM ( ' + query + ') AS query;'
    # print('querying size...')
    # size = np.array(db_query(db_object, s_query, None, False)).ravel()[0]

    # return size
    lim_arg = ':limit'
    off_arg = ':offset'

    if query.find(':limit') < 0 and query.find(':offset') < 0:
        lim_arg = ':_limit'
        off_arg = ':_offset'
        query = 'SELECT * FROM ( ' + query + ') AS query LIMIT :_limit OFFSET :_offset;'

    if arguments is None:
        arguments = dict()
    arguments[lim_arg] = partition

    offset = 0
    total_rows = 0
    inc = 1

    output_str = '-- Part {} Retrieved ({} Total Rows)'.format(0, 0)
    sys.stdout.write(output_str)
    sys.stdout.flush()
    old_str = output_str

    while True:
        arguments[off_arg] = offset
        res, hdr = db_query_fetch(db_object, query, arguments, True)

        total_rows += len(res)

        if len(res) == 0:
            break

        if len(res) > 0:
            write_csv(res,csv_out_filename,hdr if offset == 0 else None, offset != 0)

        sys.stdout.write('\r' + (' ' * len(old_str)))
        output_str = '\r-- Part {} Retrieved ({} Total Rows)'.format(inc, total_rows)
        sys.stdout.write(output_str)
        sys.stdout.flush()
        old_str = output_str

        del res

        offset += partition
        inc += 1

    sys.stdout.write('\r' + (' ' * len(old_str)))
    output_str = '\r-- {} Total Rows Retrieved ({} Parts)'.format(total_rows, inc)
    sys.stdout.write(output_str)
    sys.stdout.flush()

    return total_rows





class TableBuilder:
    def __init__(self, name):
        self.fields = []
        self.name = name
        self.num_fields = 0
        self.__primary = False

    def add_field(self, name, type, primary=False):
        if primary:
            assert not self.__primary
            self.__primary = True

        assert type in ['integer', 'bigint', 'double precision', 'text', 'timestamp'] # limited data type support
        self.fields.append({'name': name.replace(' ', '_') if not primary else 'id',
                            'type': type if not primary else 'bigint',
                            'primary': primary,
                            'values': []})
        self.num_fields += 1

        return self

    def get_fields(self, as_string=False):
        fields = [f['name'] for f in self.fields]
        if not as_string:
            return fields

        f_str = fields[0]
        for f in range(1,len(fields)):
            f_str += ', ' + fields[f]
        return f_str


def db_write_from_csv(filename, db_object, table=None, primary_column=None):
    data, header = read_csv(filename, 100)
    data = np.array(data)
    header = np.array(header)

    primary = primary_column
    if primary_column is None:
        data = np.insert(data,0,range(len(data)),1)
        header = np.insert(header,0,'id')
        primary = 0

    tname = filename[0:-4]
    if table is not None and table.num_fields == 0:
        tname = table.name

    if table is None:
        table = TableBuilder(tname)

    if table.num_fields == 0:
        for i in range(len(header)):
            table.add_field(header[i],infer_basic_type(data.T[i],100),i==primary)

    query = 'DROP TABLE IF EXISTS ' + table.name + ';\n'
    query += 'CREATE TABLE ' + table.name + '('

    for f in range(table.num_fields):
        query += table.fields[f]['name'] + ' ' + table.fields[f]['type'] + ' '
        if table.fields[f]['primary']:
            query += 'PRIMARY KEY'

        if f < table.num_fields - 1:
            query += ', '
        else:
            query += '); '
    print('-- creating table: {}'.format(table.name))
    db_query(db_object, query)

    query = 'INSERT INTO ' + table.name + ' VALUES '

    n_lines = 0

    with open(filename, 'r', errors='ignore') as f:
        f_lines = f.readlines()
        n_lines = len(f_lines)
        output_str = '-- loading {}...({}%)'.format(filename, 0)
        sys.stdout.write(output_str)
        sys.stdout.flush()
        old_str = output_str

        for i in range(1, n_lines):
            line = f_lines[i].strip()

            ind = line.find('\"')
            while not ind == -1:
                end = line.find('\"', ind+1)
                if end == -1:
                    break
                comma = line.find(',',ind, end)
                while not comma == -1:
                    line = line[:comma] + '<comma>' + line[comma + 1:]
                    end = line.find('\"', ind + 1)
                    comma = line.find(',', comma, end)
                ind = line.find('\"', end+1)

            apostrophe = line.find('\'')
            while not apostrophe == -1:
                line = line[:apostrophe] + '<apostrophe>' + line[apostrophe + 1:]
                apostrophe = line.find('\'', apostrophe+len('<apostrophe>'))

            csvalues = np.array(line.replace('\"', '').split(','))

            val = '('

            if primary_column is None:
                val += str(i)
                if len(csvalues) > 0:
                    val += ', '

            for j in range(len(csvalues)):
                csvalues[j] = csvalues[j].replace('<comma>', ',')
                csvalues[j] = csvalues[j].replace('<apostrophe>', '\'\'')

                if table.num_fields == len(csvalues) + (1 if primary_column is None else 0) \
                        and table.fields[j + (1 if primary_column is None else 0)]['type'] in ['text', 'timestamp']:
                    val += '\'' + csvalues[j] + '\''
                elif csvalues[j] == '' or csvalues[j] is None:
                    val += 'NULL'
                else:
                    val += csvalues[j]
                if j < len(csvalues) - 1:
                    val += ', '
            val += ')'

            query += val
            if not round((i/n_lines)*100, 2) == round(((i-1)/n_lines)*100, 2):
                sys.stdout.write('\r' + (' ' * len(old_str)))
                output_str = '\r-- loading {}...({}%)'.format(filename, round((i / n_lines) * 100, 2))
                sys.stdout.write(output_str)
                sys.stdout.flush()
                old_str = output_str

                query += ';'
                db_query(db_object, query)
                query = 'INSERT INTO ' + table.name + ' VALUES '
            else:
                query += ', ' if i < n_lines - 1 else ';'
        sys.stdout.write('\r' + (' ' * len(old_str)))
        output_str = '\r-- loading {}...({}%)'.format(filename, 100)
        sys.stdout.write(output_str)
        sys.stdout.flush()

        if query[-1] == ';':
            db_query(db_object, query)
    print('-- {} rows inserted into {}'.format(n_lines, table.name))


def db_create_table(db_object, table):
    assert table.num_fields > 0
    query = 'DROP TABLE IF EXISTS ' + table.name + ';\n'
    query += 'CREATE TABLE ' + table.name + '('

    for f in range(table.num_fields):
        query += table.fields[f]['name'] + ' ' + table.fields[f]['type'] + ' '
        if table.fields[f]['primary']:
            query += 'PRIMARY KEY'

        if f < table.num_fields - 1:
            query += ', '
        else:
            query += '); '
    print('-- creating table: {}'.format(table.name))
    db_query(db_object, query)


def db_write(df, db_object, table, append=False):
    assert type(table) is TableBuilder

    if not append:
        db_create_table(db_object, table)

    df = np.array(df)
    assert len(df.shape) in [1,2]

    if len(df.shape) == 1:
        df = df.reshape((-1,1))

    is_str = [infer_if_string(df[:, i], 100) for i in range(df.shape[1])]

    query = 'INSERT INTO ' + table.name + ' VALUES '
    for i in range(df.shape[0]):
        line = '('
        for j in range(df.shape[1]):
            if (table.num_fields == df.shape[1] and table.fields[j]['type'] in ['text', 'timestamp']) or is_str[j]:
                line += '\'' + df[i,j] + '\''
            elif df[i,j] == '' or df[i,j] is None:
                line += 'NULL'
            else:
                line += df[i,j]
            if j < df.shape[1]-1:
                line += ', '
        line += ')'
        line += ', ' if i < df.shape[0]-1 else ';'
        query += line

    db_query(db_object, query)
    print('-- {} rows inserted into {}'.format(df.shape[0],table.name))


def median_absolute_deviation(ar):
    ar = np.array(ar, dtype=np.float32)

    med = np.nanmedian(ar)
    return np.nanmean(np.abs(ar-med))


class Split:
    MEAN = 'mean'
    MEDIAN = 'median'
    MIN = 'min'
    MAX = 'max'


class Transforms:
    NONE = 'identity'
    COPY = 'copy'
    LOG = 'log'
    CORRECTED_LOG = 'corrected_log'
    ROOT = 'root'
    ZSCORE = 'zscore'
    MZSCORE = 'mod_zscore'
    WINSORIZE = 'winsor'
    DICHOTOMIZE = 'bin'
    OFFSET = 'offset'
    INVERSE = 'inverse'

    @staticmethod
    def copy(df, column):
        return np.array(df)[:, column]

    @staticmethod
    def log(df, column):
        cov = as_float(df[:, column])
        return np.log(cov)

    @staticmethod
    def corrected_log(df, column):
        cov = as_float(df[:, column])
        m = np.nanmin(cov)
        if m < 1:
            cov = (cov - m) + 1
        return np.log(cov)

    @staticmethod
    def root(df, column):
        cov = as_float(df[:, column])
        return np.sqrt(cov)

    @staticmethod
    def zscore(df, column):
        cov = as_float(df[:, column])
        return (cov-np.nanmean(cov))/np.nanstd(cov)

    @staticmethod
    def modified_zscore(df, column):
        cov = as_float(df[:, column])
        MAD = median_absolute_deviation(cov)
        const = 1.253314 if MAD == 0 else 1.486
        return (cov - np.nanmedian(cov)) / (MAD * const)

    @staticmethod
    def winsorize(df, column, min=None, max=None):
        cov = as_float(df[:, column])
        if min is not None:
            cov[np.argwhere(np.array(cov[:]) < min)] = min
        if max is not None:
            cov[np.argwhere(np.array(cov[:]) > max)] = max
        return cov

    @staticmethod
    def dichotomize(df, column, split):
        cov = as_float(df[:, column])

        val = None
        try:
            val = float(split)
        except ValueError:
            if split == Split.MEAN:
                val = np.nanmean(cov)
            elif split == Split.MEDIAN:
                val = np.nanmedian(cov)
            elif split == Split.MIN:
                val = np.nanmin(cov)
            elif split == Split.MAX:
                val = np.nanmax(cov)

        if val is None:
            raise ValueError('Unrecognized split method')

        nan = np.argwhere(np.array(cov,dtype=str) == 'nan').ravel()
        cov = np.array(cov == val, dtype=np.float32)
        if len(nan) > 0:
            cov[nan] = float('nan')
        return cov

    @staticmethod
    def offset(df, column, value):
        cov = as_float(df[:, column]) + value
        return cov

    @staticmethod
    def inverse(df, column, split):
        cov = as_float(df[:, column])

        val = None
        try:
            val = float(split)
        except ValueError:
            if split == Split.MEAN:
                val = np.nanmean(cov)
            elif split == Split.MEDIAN:
                val = np.nanmedian(cov)
            elif split == Split.MIN:
                val = np.nanmin(cov)
            elif split == Split.MAX:
                val = np.nanmax(cov)

        if val is None:
            raise ValueError('Unrecognized split method')

        nan = np.argwhere(np.array(cov, dtype=str) == 'nan').ravel()
        dif = np.array(cov - val, dtype=np.float32)
        cov = (np.ones(cov.shape) * val) - dif

        if len(nan) > 0:
            cov[nan] = float('nan')
        return cov

    def __init__(self, df, headers):
        self.__transforms = {
            'copy': Transforms.copy,
            'log': Transforms.log,
            'corrected_log': Transforms.corrected_log,
            'root': Transforms.root,
            'zscore': Transforms.zscore,
            'mod_zscore': Transforms.modified_zscore,
            'winsor': Transforms.winsorize,
            'bin': Transforms.dichotomize,
            'offset': Transforms.offset,
            'inverse': Transforms.inverse,
        }

        self.__df = np.array(df)
        self.__headers = np.array(headers, dtype=np.dtype('U255'))
        assert (len(df.shape) == 2 and len(headers.shape) == 1 and df.shape[1] == len(headers))

        self.__modified = []

    def apply(self, transform, column, **args):
        if transform == Transforms.NONE:
            return self

        feed_dict = {'df': self.__df, 'column': column}

        if column == -1:
            self.__modified[-1] = self.__df.shape[1]
        else:
            self.__modified.append(self.__df.shape[1])

        for i in args:
            feed_dict[i] = args[i]

        self.__df = np.insert(self.__df, self.__df.shape[1], self.__transforms[transform](**feed_dict), axis=1)
        self.__headers = np.insert(self.__headers, len(self.__headers),
                            str(transform) + '_' + str(np.array(self.__headers[column]))).reshape((-1))
        return self

    def get(self):
        return self.__df, self.__headers

    def get_modified_columns(self):
        return self.__modified


if __name__ == "__main__":

    data = [[0,1,2,'apple'],[0,1,1,'abble'],[1,1,1,'apple'],[1,1,2,'apple'],[0,2,1,'appge'],[0,2,2,'appple']]

    print(levenshtein_ratio(data,3,[0,1],2))
    exit(1)

    data, headers = read_csv('resources/DKT_test.csv')
    print_descriptives(data,headers)

    cf = cross_feature(data[:,1],data[:,3],np.nan, distinct_feature_values=[0,1])
    for i in cf:
        print(i)
        print(cf[i])
