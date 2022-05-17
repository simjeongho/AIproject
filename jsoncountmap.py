import numpy as np
import openpyxl

wb = openpyxl.load_workbook("jsontest.xlsx", read_only=False, data_only=False)
ws = wb['Sheet1']

count = np.unique(var, return_counts=True)
print(count[0])
wb.save('jsontest1.xlsx')
