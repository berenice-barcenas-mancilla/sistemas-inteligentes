<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    //Obtener los valores de las respuestas
    $res1 = $_POST["res1"];
    $res2 = $_POST["res2"];
    $res3 = $_POST["res3"];
    $res4 = $_POST["res4"];
    $res5 = $_POST["res5"];
    $res6 = $_POST["res6"];

    //Puntaje total
    $PT = $res1 + $res2 + $res3 + $res4 + $res5 + $res6;

    //Establecer el umbral para determinar la probabilidad de depresión
    $nivelDepresion = 6;

    // Determinar la probabilidad de depresión
    if ($PT >= $nivelDepresion) {
        $resultado = "La probabilidad de tener depresión es baja.";
    } else {
        $resultado = "Existe una probabilidad alta de tener depresión.";
    }

    //Imprimir resultado
    echo "Tu puntaje total es: $PT<br>";
    echo $resultado;
}
