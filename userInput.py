
#folderPath = "C:\\Users\\User\\Desktop\\master\\"
fileName = 'UserValues.txt'

f = open(fileName, "w")

tertagonika="0"
domatia="0"
mpania="0"
orofos="0"
etos=0
parking="0"
thermansi="0"
typos="0"
endodapedia="0"

print("Please write the square meters of the house")
tertagonika=input()
f.write(tertagonika+"\n")

print("Please write the rooms of the house")
domatia=input()
f.write(domatia+"\n")

print("Please write the baths of the house")
mpania=input()
f.write(mpania+"\n")

print("Please write the floor of the house")
orofos=input()
f.write(orofos+"\n")

print("Please write the number of your choice if the house has parking\n1. Yes\n2. No")
parking=input()
if (parking=="1"):
    f.write("1"+"\n")
if (parking=="2"):
    f.write("0"+"\n")

print("Please write the number of your choice about the type of the house\n1. Apartement\n2. Maisonette\n3. Studio\n4. Single family house\n5. Building\n6. Complex")
typos=input()
if (typos=="1"):
    f.write("1"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
if (typos=="2"):
    f.write("0"+"\n")
    f.write("1"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
if (typos=="3"):
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("1"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
if (typos=="4"):
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("1"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
if (typos=="5"):
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("1"+"\n")
    f.write("0"+"\n")
if (typos=="6"):
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("1"+"\n")

print("Please write the number of your choice about the type of the house heating\n1. Auto electricity\n2. Auto oil\n3. Auto gas\n4. Central oil\n5. Central gas")
thermansi=input()
if (thermansi=="1"):
    f.write("1"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
if (thermansi=="2"):
    f.write("0"+"\n")
    f.write("1"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
if (thermansi=="3"):
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("1"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
if (thermansi=="4"):
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("1"+"\n")
    f.write("0"+"\n")
if (thermansi=="5"):
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("0"+"\n")
    f.write("1"+"\n")

print("Please write the number of your choice if the house has floor heating\n1. Yes\n2. No")
endodapedia=input()
if (endodapedia=="1"):
    f.write("1"+"\n")
if (endodapedia=="2"):
    f.write("0"+"\n")

print("Please write the year of construction of the house")
etos=input()
f.write(str(2022-int(etos)))

f.close()

