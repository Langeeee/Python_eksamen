# Python-eksamen 4 semester 2021 

---------------------
   Captcha cracker 
---------------------

1. Gruppe medlemmer: Jacob Lange Nielsen og Jonathan Juhl
  
2. Kort beskrivelse af projektet:

  I dette projekt har vi til formål at træne en maskine, til at lære at genkende tal 
  fra Captchas. Dette gøres vha. web scraping, hvorigennem vi får
  kreeret mock-captchas. Disse gemmes og herefter laver vi trænings- og kontrolsæt af af   dem.
  Yderligere vil vi bruge datasættet mnist, der består af håndskrevne tal for at se,
  hvilken af disse metoder maskinen lærer bedst af.
  
3. ressourcer:

  Captcha:
  https://fakecaptcha.com/
  Handwriting samples:
  https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist#0

4. teknologier:

  mnist
  numpy 
  pandas
  keras
  matplotlib
  cv2
  tenserflow
  selenium
  base64
  re
  easyocr
5. installations guide: 

  install the following teknoligies:
    numpy 1.19.2
    tenserflow 2.6.0
    selenium 3.141.0
    keras 2.6.0
    pandas 1.0.4
    easyocr 1.4.1
    
6. Bruger guide 
    - Hent og installer alle teknologier beskrevet ovenfor via “pip install …”
    - Programmet køres via IPYNB/TestMachnies
	- Heri kan maskiner trænes og gemmes med forskellige datasæt og præcision kan hermefter aflæses
	- Mnist maskinen, kan gætte på billeder fra mappen med beskårne billeder
	- I det nederste felt trænes 2 maskiner med hvert deres træningssæt, men med det samme test-sæt og der displayes grafer over præcisionen af disse. 

   - I IPYNB ligger også “Untitled.IPYNB” hvori billederne bliver beskåret, kør dog denne varsomt, da den kan chrashe ens pc ( yderligere er alle billederne allerede beskårne og denne behøver derfor ikke køres)

I Webscraping/webscraping.ipynb kan der laves websraping af sitet fakecapctha, som via en tilfældig streng af 4 tal mellem 0-9 downloader og gemmer de genererede billeder. 
   
7. Status på projekt

Projektet har diverse fejl og mangler, men er alt i alt temmeligt funktionelt. Projektet kan indhendte billeder via webscraping, beskære disse vha. contouring og derefter gemme dem i passende foldere. Herefter kan der via keras laves dataset ud fra disse foldere, som maskinerne kan trænes med. Der er lavet 2 forskellige dataset, et “verificeret” hvori vi selv har downloaded billeder og verificeret at der ikke er fejl på, førend de blev beskåret og et andet, hvor billederne stammer fra webscraping og er uverificeret korrekte, men derimod talstærke. Overordnet set kan det konkluderes ud fra vores grafer, at selvom der kan være fejl på nogle af billederne i det “uverficerede” datasæt er dette at foretrække da vi ender op med en højere præcision i sidste ende. Samtidigt har vi trænet en maskine med “mnist” datasættet, for at se, om håndskrevne tal er sammenlignelige med “captchas”, hvilket umiddelbart ikke er tilfældet. Da vi fik et “shape mismatch” da vi forsøgte at træne denne med vores eget kreerede test-datasæt, valgte vi at få maskinen til at gætte på hvert eneste billede i stedet og manuelt lave en optælling af hvor mange den ramte rigtigt. Det må konkluderes at en maskine der er trænet med det rigtige datasæt er langt mere præcis, selvom håndskrevne tal og tal fra en “captcha”  for det menneskelige øje virker meget sammenlignelig. Vi har også forsøgt at analysere sammenhængen mellem antallet af epoker i en machine learning algoritme og størrelsen på et dataset. Hvis man kigger på det sidste felt i “TestMachines.IPYNB” kan det ses at mnist datasættet faktisk er bedre til at gætte end det verificerede, såfremt de får lov at træne lige mange gange på deres datasæt. 










8. Udfordringer:

Der er spørgsmålstegn i webscrapede captcha's i stedet for 0 og 1 i nogle tilfælde, hvilket dsv gør vores datasæt mere upræcise.

Problem med cutting af for lange captchas.Så vi var nødsaget til at wepscrape flere  captchas der var kortere(4 tegn i stedet for 10). Dette grundet overlappende tal, som cv2.Contour ikke kunne finde ud af at adskille.

Forskellig opsætning af dataformat gav problemer da maskinerne skulle trænes og testes

Under webscraping var vi nødsaget til at tage billedet’s source og base64 decode det, da download knappen på fakecaptcha er defekt. Yderligere blev brugt en del tid, på at finde ud af, at den URL der stod i sourcen, ikke var en URL der kunne downloades, men derimod et billede i base64 encoding. 

Det var en udfordring at kreere et datasæt i passende størrelse med verificerede værdier. 
