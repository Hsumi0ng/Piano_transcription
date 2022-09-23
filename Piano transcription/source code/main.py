
from get_keyboard import *
from detectkey import *
from music21 import *
from Library import *
import time as t

tempo="50"
beat="4"
BeatPerBar="8"

TimeSignature=BeatPerBar+"/"+beat
measure=" "
for i,items in enumerate(time):
    if i==len(time)-1:break
    time[i]=time[i+1]-time[i]
time_start=t.time()
beats=change2beats(time,tempo,beat)
measure,NChord,Chordstore,Chordtime=makebar(measure,dnotes,beats)#measure是音符的记录（字符串），Nchord（记录和弦数量），Chordstore（存储和弦音符），Chordtime（存和弦持续时间）
bar = converter.parse('tinynotation: '+TimeSignature+measure) #括号内输出这个（tinynotation: 4/4 e4 e4 f4 g4 g4 f4 e4 d4 c4 c4 d4 e4 e.4 d8 d2 c=id0 c=id1 c=id2）
for i in range(NChord): #tinynotation是关键字，不用管。 meter 4/4 表示几几拍，默认为4/4，是手动修改的量，不用管。后面就是measure输出的内容。
    n = bar.recurse().getElementById('id'+str(i)) #用id标注，将占位的单音替换为和弦。
    Ingredients,d=makechord(Chordstore[i],Chordtime[i])
    ch = chord.Chord(Ingredients,duration=duration.Duration(Chordtime[i])) #Chord duration changes here
    n.activeSite.replace(n, ch)
time_end=t.time()
savepath='C:/Users/PC/Desktop/xml/'
bar.write('xml',fp=savepath+'bar.xml')
print('time cost',time_end-time_start,'s')
print('tinynotation: '+TimeSignature+measure)

