package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/jchalupk/neural/neuralnet"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type dataInput struct {
	inputs     *mat.Dense
	labels     *mat.Dense
	testInputs *mat.Dense
	testLabels *mat.Dense
}

func getInputAndLabels(fileName string) (inputs *mat.Dense, labels *mat.Dense) {
	f, err := os.Open(fileName)

	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	reader := csv.NewReader(f)

	reader.FieldsPerRecord = 7

	// Read in the csv
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	var inputsIndex int
	var labelsIndex int

	for idx, record := range rawCSVData {
		// Skip the header
		if idx == 0 {
			continue
		}

		for i, val := range record {
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	inputs = mat.NewDense(len(rawCSVData), 4, inputsData)
	labels = mat.NewDense(len(rawCSVData), 3, labelsData)

	return inputs, labels
}

func getInputData() (*dataInput, error) {

	inputs, labels := getInputAndLabels("data/train.csv")
	testInputs, testLabels := getInputAndLabels("data/test.csv")

	input := &dataInput{
		inputs:     inputs,
		labels:     labels,
		testInputs: testInputs,
		testLabels: testLabels,
	}
	return input, nil
}

func main() {
	input, err := getInputData()
	if err != nil {
		log.Fatal(err)
	}
	inputs := input.inputs
	labels := input.labels
	testInputs := input.testInputs
	testLabels := input.testLabels

	config := neuralnet.Config{
		InputNeurons:  4,
		OutputNeurons: 3,
		HiddenNeurons: 100,
		NumEpochs:     5000,
		LearningRate:  0.3,
	}

	// Create the network
	network := neuralnet.NewNetwork(config)

	// Train the network
	err = network.Train(inputs, labels)

	if err != nil {
		log.Fatal(err)
	}

	// Make some predictions
	predictions, err := network.Predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	var truePosNeg int
	numPreds, _ := predictions.Dims()
	for i := 0; i < numPreds; i++ {
		labelRow := mat.Row(nil, i, testLabels)
		var species int
		for idx, label := range labelRow {
			if label == 1.0 {
				species = idx
				break
			}
		}

		if predictions.At(i, species) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}

	// Calculate the accuracy (subset accuracy)
	accuracy := float64(truePosNeg) / float64(numPreds)

	// Output the Accuracy value to standard out.
	fmt.Printf("Accuraccy = %0.2f\n\n", accuracy)
}
