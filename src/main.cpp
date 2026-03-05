
cv::Mat stackImages(float scale, const std::vector<std::vector<cv::Mat>> &images)
{
  if (images.empty())
  {
    return cv::Mat();
  }

  std::vector<cv::Mat> stackedRows;
  stackedRows.reserve(images.size());

  int targetRows = images[0][0].rows;

  for (const auto &row : images)
  {
    if (row.empty())
    {
      continue;
    }
    
    std::vector<cv::Mat> stackedCols;
    stackedCols.reserve(row.size());
    
    for (const cv::Mat &img : row)
    {
      cv::Mat temp;
      
      if (img.rows != targetRows)
      {
        float aspect = (float)img.cols / (float)img.rows;
        cv::resize(img, temp, cv::Size(static_cast<int>(targetRows * aspect), targetRows));
      }
      else
      {
        temp = img;
      }

      if (temp.channels() == 1)
      {
        cv::cvtColor(temp, temp, cv::COLOR_GRAY2BGR);
      }

      stackedCols.push_back(temp);
    }

    cv::Mat hor;
    cv::hconcat(stackedCols, hor);
    stackedRows.push_back(hor);
  }
  
  cv::Mat outputImage;
  cv::vconcat(stackedRows, outputImage);

  if (scale != 1.0f)
  {
    cv::resize(outputImage, outputImage, cv::Size(), scale, scale);
  }

  return outputImage;
}

void getContour(cv::InputArray img, cv::Mat &imgContour)
{
  std::vector<cv::Mat> contours;
  cv::Mat hierarchy;
  cv::findContours(img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  for (cv::Mat &cnt : contours)
  {
    double area = cv::contourArea(cnt);
    if (area > 500)
    {
      cv::drawContours(imgContour, cnt, -1, cv::Scalar(255, 0, 0), 3);

      double peri = cv::arcLength(cnt, true);

      cv::Mat approx;
      cv::approxPolyDP(cnt, approx, 0.02 * peri, true);
      
      auto objCor = approx.size();

      std::string objType = ""; 
      cv::Rect box = cv::boundingRect(approx);

      if (objCor.height == 3)
      {
        objType = "Tri";
      }
      else if (objCor.height == 4)
      {
        float aspectRatio = (float) box.width / (float) box.height;
        if (aspectRatio > 0.95 && aspectRatio < 1.05)
        {
          objType = "Sqr";
        }
        else 
        {
          objType = "Rect";
        }
      }
      else 
      {
        objType = "Circle";
      }

      cv::rectangle(imgContour, box, cv::Scalar(0, 255, 0), 2);
      cv::putText(imgContour, objType, 
                  {box.x + box.width / 2 - 10, box.y + box.height / 2 - 10}, 
                  cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 0), 2
                  );
    }
  }
}

int main()
{
  cv::Mat img = cv::imread("assets/shapes.png");
  if (img.empty())
  {
    std::cout<<"Failed to open the original Image"<<std::endl;
    return -1;
  }

  cv::Mat imgContour = img.clone();

  cv::Mat imgGray;
  cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

  cv::Mat imgBlur;
  cv::GaussianBlur(imgGray, imgBlur, cv::Size(7, 7), 1);

  cv::Mat imgCanny;
  cv::Canny(imgBlur, imgCanny, 50, 50);

  cv::Mat imgBlack = cv::Mat::zeros(img.rows, img.cols, img.type());

  getContour(imgCanny, imgContour);

  cv::Mat imgStack = stackImages(0.6, 
                                 {
                                 {img, imgGray, imgBlur},
                                 {imgCanny, imgContour, imgBlack},
                                 }
  );
  
  cv::imshow("Image", imgStack);

  cv::waitKey(0);

  return 0;
}

