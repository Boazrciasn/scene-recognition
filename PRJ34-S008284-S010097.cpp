/*
  Muhammet Pakyürek	S008284 Department of Electrical and Electronics
  Baris Ozcan	S010097	Department of Computer Science
*/

#include <iostream>
#include "vector"
#include "opencv2/core.hpp"
#include "ImageReader.h"
#include "TinyImages.h"
#include "BagOfSIFT.h"
#include "KNNTest.h"
#include "SVMAnalysis.h"


using namespace std;

int main() {
    cout << "Hello, World!" << endl;

    ImageReader ImageRead;

    TinyImages TinyImages(&ImageRead);
    BagOfSIFT BagOfSIFT(&ImageRead);
    KNNTest KNNTest(&BagOfSIFT);
    SVMAnalysis SVMAnalysis(&BagOfSIFT);



    return 0;
}
