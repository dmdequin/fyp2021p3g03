import os

def __main__():
	with open("new_csv.csv", 'w') as outfile:
		for i in range(1,27):
			file = open(f"symmetry_{i}.csv", 'r')
			for line in file.readlines()[1:]:
				outfile.write(line)

	check = []
	file1 = open("new_csv.csv", 'r')
	file2 = open("symmetry.csv", 'r')
	file2.readline()
	for i in range(1933):
		check.append(file1.readline().split(',')[0] == file2.readline().split(',')[0])

	#print(file1.readline().split(',')[0])
	print(sum(check) / 1932)

	file1.close()
	file2.close()
if __name__ == '__main__':
	__main__()