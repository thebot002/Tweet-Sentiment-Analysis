import collections, csv

with open('../Data/tweets.csv', mode='r') as f:
    csvreader = csv.reader(f, delimiter=',', quotechar='\"')
    files = collections.defaultdict(lambda: [])
    for line in csvreader:
        try:
            files[int(line[0])].append(line[-1])
        except:
            print('oops')

for file in files:
    with open('../Data/tweets_{}.csv'.format(file), mode='w') as f:
        # csvwriter = csv.writer(f, delimiter=',', quotechar='\"')
        # csvwriter.writerows(files[file])
        f.writelines(files[file])

print('Success')
