import collections, csv

with open('..\\tweets.csv', mode='r') as f:
    csvreader = csv.reader(f, delimiter=',', quotechar='\"')
    files = collections.defaultdict(lambda: [])
    for line in csvreader:
        try:
            files[int(line[0])].append(''.join(line))
        except:
            print('oops')

for file in files:
    with open('../tweets_{}.csv'.format(file), mode='w') as f:
        f.writelines(files[file])

print('Success')
