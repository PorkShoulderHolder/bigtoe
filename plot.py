import matplotlib.pyplot as plt
import csv


exo_valid = []
with open('exog_results_2.txt') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in [r for r in reader][1:]:
        if len(row) > 0 and 'valid' in row[0]:
            exo_valid.append(row[1])

exo_valid = exo_valid[1:]


nexo_valid = []
with open('noexog_results_2.txt') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in [r for r in reader][1:]:
        if len(row) > 0 and 'valid' in row[0]:
            nexo_valid.append(row[1])

nexo_valid = nexo_valid[1:]

print nexo_valid
print exo_valid

plt.plot(xrange(len(nexo_valid)), nexo_valid)

plt.plot(xrange(len(exo_valid)), exo_valid, 'k--')
plt.ylim([18,50])
plt.show()
