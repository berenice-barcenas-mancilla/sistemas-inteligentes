<?php
//Comprobando si la solicitud es de tipo POST
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $numEventos = intval($_POST["numEventos"]);     //Obteniendo el número de eventos del formulario y convirtiéndolo a entero
    $probabilidades = [];     //Creando un arreglo para almacenar las probabilidades de los eventos

    //Recorriendo los eventos para obtener sus probabilidades
    for ($i = 1; $i <= $numEventos; $i++) {
        //Obteniendo la probabilidad del evento i del formulario y convirtiéndola a punto flotante
        $probabilidades[$i] = floatval($_POST["probA$i"]);
    }

    $probabilidadOtros = 1 - array_sum($probabilidades);    //Calculando la probabilidad de otros eventos (no especificados) como la diferencia entre 1 y la suma de las probabilidades ingresadas


    //Creando un array para almacenar las probabilidades de cada evento dado que ha ocurrido un evento B
    $probabilidadEventoDadoB = [];
    //Recorriendo los eventos para obtener las probabilidades condicionales P(B|Ai)
    for ($i = 1; $i <= $numEventos; $i++) {
        //Obteniendo la probabilidad de B dado Ai del formulario y convirtiéndola a punto flotante
        $probabilidadEventoDadoB[$i] = floatval($_POST["probB_given_A$i"]);
    }

    //Usando la regla de Bayes para calcular P(Ai|B), la probabilidad de cada evento dado que ha ocurrido un evento B
    $probabilidadesResultantes = [];
    for ($i = 1; $i <= $numEventos; $i++) {
        //Calculando el numerador de la fórmula de Bayes (P(Ai) * P(B|Ai))
        $numerador = $probabilidades[$i] * $probabilidadEventoDadoB[$i];
        //Calculando el denominador de la fórmula de Bayes
        $denominador = $numerador;
        //Calculando la suma del numerador para todos los eventos
        for ($j = 1; $j <= $numEventos; $j++) {
            //Excluyendo el evento i actual del cálculo del denominador
            if ($j !== $i) {
                $denominador += $probabilidades[$j] * $probabilidadEventoDadoB[$j];
            }
        }
        //Calculando la probabilidad condicional P(Ai|B) y almacenándola en el array de resultados
        $probabilidadesResultantes[$i] = $numerador / $denominador;
    }

    //Construyendo la respuesta final
    $respuesta = "";
    for ($i = 1; $i <= $numEventos; $i++) {
        // Concatenando la probabilidad condicional de cada evento con su respectivo mensaje
        $respuesta .= "La probabilidad de que ocurra el evento $i dado que ha ocurrido un evento B es: " . $probabilidadesResultantes[$i] . "\n";
    }
    // Imprimiendo la respuesta
    echo $respuesta;
}
