# GLAD SOM
##Fazy algorytmu
SOM składa się z epok. Podczas każdej epoki wszystkie zamówienia są prezentowaniu rozwiązaniu w losowej kolejności.
Dla każdego zamówienia jest znajdowany "najlepszy"  wierzchołek. Następnie znaleziony wierzchołek wraz z otoczeniem
jest przesuwany w kierunku prezentowanego zamówienia.  
  
Podczas każdej epoki mogą zostac wykonane dodatkowe operacje:  
Blokowanie przeładowanych tras - wierzchołki z zablokowanych tras nie będą mogły być wybrane jako 
najlepsze przez zadaną ilość epok   
Wyrównywanie wszytkich tras - cofamy wierzchołki na trasach które nie były ruszane przez ustaloną ilość epok

##Parametry
Wszystkie stałe są ustawiane w knstruktorze configu. Najważniejsze stałe to:  
mi - Stała ruchu w kierunku wybranego wierzchołka  
lambda - Stała ruchu w kierunku sąsiadów  
expected_ratio - Stosunek "ważności" odległości i capacity przy wyborze wierzchołka na trasie dla danego zamówienia 
(im wyższy tym bardziej preferujemy bliskie trasy, im niższy tym bardziej preferujemy trasy o małym załadowaniu)  
learning_rate - Każdy wektor przesunięcia jest przez przez nią przemnożony  
blocking_frequency - Jak często blokować przeładowane trasy  
blocking_period - Na jak długo zablokować przeładowane trasy  
F_neurons_percentage - Określi jaki fragment pierścienia porusza się w kierunku zamówienia  
G_neurons_percentage - Gradientuje ruch w kierunku zamówienia na fragmencie pierścienia, im większe G tym tuch jest "ostrzejszy"
    
 
  