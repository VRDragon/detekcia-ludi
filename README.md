# 🧍 PeopleSpotter

**Počítač, ktorý sa pozerá na záznam z kamery a zapisuje, kedy sa v zábere objavili ľudia.**

![Rok](https://img.shields.io/badge/rok-2017-blue) ![Python](https://img.shields.io/badge/Python-2.7-yellow) ![OpenCV](https://img.shields.io/badge/OpenCV-2.4-green) ![Stav](https://img.shields.io/badge/stav-archív-lightgrey)

> **Rok realizácie:** 2017 · **Tagy:** `opencv` `python` `computer-vision` `haar-cascade` `people-detection` `video-analysis` `student-project`

## O projekte

PeopleSpotter je experiment z roku 2017 s klasickým počítačovým videním (bez neurónových sietí). Skript prechádza video snímku po snímke, hľadá v nich hornú polovicu postavy pomocou **Haarovej kaskády** (`haarcascade_mcs_upperbody.xml`) a do konzoly vypisuje, od akého času bol v zábere nejaký človek a kedy zmizol.

Predvolený vstup je nahrávka z kamery `md_stud_cam2_201702222000.mp4` (názov naznačuje záznam z 22. 2. 2017 o 20:00). Namiesto videa možno zadať aj živú kameru.

## Ako to funguje

1. Snímka sa prevedie na odtiene sivej a vyrovná sa jej histogram (`equalizeHist`), aby bola detekcia odolnejšia voči svetlu.
2. `detectMultiScale` nájde obdĺžniky s horným telom (min. 30 × 30 px).
3. Každá detekcia sa vykreslí zeleným rámikom, v jej vnútri sa ešte hľadá dolná časť tela (modrý rámik).
4. Ak sa na jednu sekundu videa (podľa FPS) nič nenájde, vypíše sa čas odchodu: `do <čas>`.
5. Pri nájdení osôb sa vypíše `N osôb na obzore od <čas>`.

## Spustenie

Projekt je písaný pre **Python 2.7 a OpenCV 2.4** (používa rozhranie `cv2.cv`) a využíva pomocné moduly `video.py` a `common.py` z ukážok OpenCV.

```bash
python peopledetect.py [--cascade haarcascade_mcs_upperbody.xml] [video.mp4]
```

`ESC` ukončí program.

## Poznámky

- Moderné OpenCV (3/4) túto verziu kódu nespustí bez úprav (`cv2.cv` už neexistuje).
- Cesta ku kaskáde dolného tela (`../../data/haarcascades/...`) je pevne daná a v repozitári nie je.
- Haarova kaskáda dáva falošné detekcie; dnes by sa použil HOG+SVM alebo model typu YOLO.

## Autor

**Vratislav Režo** ([@VRDragon](https://github.com/VRDragon))
