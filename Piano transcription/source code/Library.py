import cv2
import numpy as np
from skimage import measure
import matplotlib.patches as mpatches

#initial parameters
# music data structure dictionary<->hash list o(1)
#pitch & duration & tempo & meter & measure
# CCC=C1,CC=C2,C=C3,c=C4,c'=C5,c''=C6 c#/c-=C sharp/C flat C[4,8,16,32]means duration

notes={0:"AAAA",1:"BBBB",2:"CCC",3:"DDD",4:"EEE",5:"FFF",6:"GGG",
       7:"AAA",8:"BBB",9:"CC",10:"DD",11:"EE",12:"FF",13:"GG",
       14:"AA",15:"BB",16:"C",17:"D",18:"E",19:"F",20:"G",
       21:"A",22:"B",23:"c",24:"d",25:"e",26:"f",27:"g",
       28:"a",29:"b",30:"c'",31:"d'",32:"e'",33:"f'",34:"g'",
       35:"a'",36:"b'",37:"c''",38:"d''",39:"e''",40:"f''",41:"g''",
       42:"a''",43:"b''",44:"c'''",45:"d'''",46:"e'''",47:"f'''",48:"g'''",
       49:"a'''",50:"b'''",51:"c''''",
       52:"AAAA#",53:"CCC#",54:"DDD#",55:"FFF#",56:"GGG#",
       57:"AAA#",58:"CC#",59:"DD#",60:"FF#",61:"GG#",
       62:"AA#",63:"C#",64:"D#",65:"F#",66:"G#",
       67:"A#",68:"c#",69:"d#",70:"f#",71:"g#",
       72:"a#",73:"c'#",74:"d'#",75:"f'#",76:"g'#",
       77:"a'#",78:"c''#",79:"d''#",80:"f''#",81:"g''#",
       82:"a''#",83:"c'''#",84:"d'''#",85:"f'''#",86:"g'''#",87:"a'''#"
       }

durations={0.25:"16",0.375:".16",0.5:"8",0.75:".8",1:"4",1.5:".4",2:'2',3:'.2',4:'1'}

tempos={'adagio':56,'moderato':96,'allergo':132}

def change2beats(time,tempo,beat):
    d=[0,0.25,0.375,0.5,0.75,1,1.5,2,3,4] #时值数组 比大小防过界 加了“0”
    if tempo in tempos:
        tempo=tempos[tempo]
    else:tempo=int(tempo)
    reverse = {v : k for k, v in durations.items()}
    beat=reverse[beat] #beat 获取 对应 时值
    time=np.array(time)
    time=0.01667*time*tempo  # 时间->拍数              #四舍五入
    beats=beat*time                 #对应拍数
    seq=[]
    for item in beats:
       seq.append(np.where(d<item)[-1][-1])
    seq=np.array(seq)
    seq[seq==0]=1
    for i,item in enumerate(seq):
        beats[i]=d[item]
    beats=beats.tolist()
    return beats


def makebar(measure,dnotes,timevalue):
    NChord=0
    Chordstore=[]
    Chordtime=[]
    for i,item in enumerate(dnotes):
        if isinstance(item, list):
            measure=measure+'c=id'+str(NChord)+' '
            Chordstore.append(item)
            Chordtime.append(timevalue[i])
            NChord=NChord+1
        else:
            measure=measure+notes[item]+durations[timevalue[i]]+' '
    return measure,NChord,Chordstore,Chordtime

def makechord(chord,Chordtime):
    Ingredients=''
    for item in chord:
        if notes[item][0] in ('A','B','C','D','E','F','G'):
            if notes[item][-1] in("#","b"):
                pitch=5-len(notes[item])
                notes[item]=notes[item][0]+str(pitch)+notes[item][-1]
            else:
                pitch=4-len(notes[item])
                notes[item]=notes[item][0]+str(pitch)
        if notes[item][-1] =="'":
                pitch=3+len(notes[item])
                notes[item]=notes[item][0]+str(pitch)
        if ord(notes[item][0])>96 and notes[item][-1] in ("#","b"):
            pitch=2+len(notes[item])
            notes[item]=notes[item][0]+str(pitch)+notes[item][-1]
        Ingredients=Ingredients+notes[item]+' '
        d=Chordtime
    return Ingredients,d
