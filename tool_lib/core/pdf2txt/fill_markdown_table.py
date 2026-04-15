import re

def extract_markdown_tables(text):
    """
    从文本中提取所有的 Markdown 表格，并返回它们的起始和结束行号以及表格内容。
    """
    lines = text.split('\n')
    tables = []
    in_table = False
    table_start = None
    table_lines = []

    for idx, line in enumerate(lines):
        if re.match(r'\s*\|', line):
            if not in_table:
                in_table = True
                table_start = idx
                table_lines = [line]
            else:
                table_lines.append(line)
        else:
            if in_table:
                in_table = False
                tables.append((table_start, idx - 1, table_lines))
                table_lines = []

    # 如果文本以表格结尾
    if in_table:
        tables.append((table_start, len(lines) - 1, table_lines))

    return tables

def parse_markdown_table(table_lines):
    """
    将表格的行解析成二维列表
    """
    rows = []
    for line in table_lines:
        # 去除首尾的空格和竖线
        line = line.strip().strip('|')
        # 忽略分隔行（例如：| --- | --- |）
        if re.match(r'^-+', line.replace(' ', '').replace('|', '')):
            continue
        # 分割单元格并去除每个单元格的空格
        cells = [cell.strip() for cell in line.split('|')]
        rows.append(cells)
    return rows

def remove_consecutive_spaces(text, n=5):
    """
    移除字符串中连续 n 个或以上的空格
    """
    return re.sub(r' {'+str(n)+',}', '', text)

def process_cell_content(cell):
    """
    处理单元格内容，移除连续5个或以上的空格
    """
    return remove_consecutive_spaces(cell, 5)

def fill_missing_cells(rows):
    header = rows[0]
    data_rows = rows[1:]

    num_columns = len(header)
    last_values = [''] * num_columns  # 初始化记录每列上一个非空值的列表

    new_rows = [header]

    for row in data_rows:
        new_row = []
        for i in range(num_columns):
            # 防止行的列数不足
            cell = row[i] if i < len(row) else ''
            # 移除连续5个或以上的空格
            cell = process_cell_content(cell)
            if cell:  # 当前单元格非空
                last_values[i] = cell  # 更新上一个非空值
            else:
                cell = last_values[i]  # 用上一个非空值填充
            new_row.append(cell)
        new_rows.append(new_row)
    return new_rows

def get_column_widths(rows):
    num_columns = len(rows[0])
    widths = [0] * num_columns
    for row in rows:
        for i in range(num_columns):
            widths[i] = max(widths[i], len(row[i]))
    return widths

def generate_markdown_table(rows):
    widths = get_column_widths(rows)
    num_columns = len(rows[0])

    # 生成表头和分隔线
    header_line = '| ' + ' | '.join(rows[0][i].ljust(widths[i]) for i in range(num_columns)) + ' |'
    separator_line = '|-' + '-|-'.join('-' * widths[i] for i in range(num_columns)) + '-|'

    table_lines = [header_line, separator_line]

    # 生成数据行
    for row in rows[1:]:
        line = '| ' + ' | '.join(row[i].ljust(widths[i]) for i in range(num_columns)) + ' |'
        table_lines.append(line)

    return table_lines

def process_text(text):
    lines = text.split('\n')
    tables = extract_markdown_tables(text)
    if not tables:
        return text  # 如果没有找到表格，返回原文本

    # 用于替换原始表格的行
    new_lines = lines.copy()

    for table_start, table_end, table_lines in tables:
        # 解析表格
        rows = parse_markdown_table(table_lines)
        # 检查是否解析成功
        if len(rows) < 2:
            continue  # 忽略只有表头或者内容不足的表格
        # 填充空单元格并处理内容
        filled_rows = fill_missing_cells(rows)
        # 生成新的表格行
        new_table_lines = generate_markdown_table(filled_rows)
        # 替换原始表格的行
        new_lines[table_start:table_end + 1] = new_table_lines

    # 将处理后的行合并为文本
    return '\n'.join(new_lines)

# 示例输入的文本
text = '''
xxxxxxxxxxxxxxx

| 制度面向  | 辦理項目                   | 辦理項目細項                                                                                                                           | 辦理內容                                                      |
|-------|------------------------|----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| 管理面   | 資通系統分級及防護基準            | 初次受核定或等級變更後之一年內，針對自行或委外開發之資通系統，依附表九完成資通系統分級，並完成附表十之控制措施；其後應每年至少檢視一次資通系統分級妥適性。                                                    |                                                           |
| 管理面   | 資訊安全管理系統之導入及通過公正第三方之驗證 | 初次受核定或等級變更後之二年內，全部核心資通系統導入     CNS 27001 或 ISO 27001 等資訊安全管理系統標準、其他具有同等或以上效果之系統或標準，或其他公務機關自行發展並經主管機關認可之標準，於三年內完成公正第三方驗證，並持續維持其驗證有效性。 |                                                           |
| 管理面   | 資通安全專責人員               | 初次受核定或等級變更後之一年內，配置四人；須以專職人員配置之。                                                                                                  |                                                           |
| 管理面   | 內部資通安全稽核               | 每年辦理二次。                                                                                                                           |

xxxxxxxxxxxxxxxxxxxxx
'''

# 处理文本
new_text = process_text(text)

# print(new_text)
