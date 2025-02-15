#include "..\FidelityFX-CAS-CUDA\include\CASLibWrapper.h"
#include "mainwindow.h"
#include "widget_utils.hpp"
#include <QFileDialog>
#include <QFileDialog>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QImage>
#include <QImageReader>
#include <QLabel>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPixmap>
#include <QScreen>
#include <QSlider>
#include <QString>
#include <QtMinMax>
#include <QVBoxLayout>
#include <QWidget>
#include <type_traits>
#include <QScrollArea>

#define CLAMP(x) qBound(0.0f, x/100.0f, 1.0f)

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    sharpenStrength(new QSlider(Qt::Horizontal)),
    contrastAdaption(new QSlider(Qt::Horizontal)),
    imageView(new ZoomableLabel),
    scrollArea(new QScrollArea),
    sharpenStrengthLabel(new QLabel("Sharpen Strength")),
    contrastAdaptionLabel(new QLabel("Contrast Adaption")),
    casObj(CAS_initialize()),
    // 80% of the screen size
    targetImageSize(QGuiApplication::primaryScreen()->availableGeometry().size() * 0.8)
{
    //setup sliders
    setupSlider(sharpenStrength, sharpenStrengthLabel, 0);
    setupSlider(contrastAdaption, contrastAdaptionLabel, 100);
    //setup file menu
    setupMenu();
    //setup main image view
    setupImageView();
    //setup main widget
    setupMainWidget();
}

//destroy DLL's memory
MainWindow::~MainWindow()
{
    CAS_destroy(casObj);
}

//setup CAS parameter sliders
void MainWindow::setupSlider(QSlider *slider, QLabel *label, const int value) const
{
    //Setup slider properties (QSlider and correspinding QLabel)
    slider->setRange(0, 100);
    slider->setValue(value);
    label->setFixedWidth(130);
    WidgetUtils::setVisibility(false, slider, label);
    connect(slider, &QSlider::valueChanged, this, &MainWindow::sliderValueChanged);
}

//setup "File" menu
void MainWindow::setupMenu()
{
    // File Menu
    QMenu *fileMenu = menuBar()->addMenu("File");
    openImageAction = fileMenu->addAction("Open Image");
    saveImageAction = fileMenu->addAction("Save Image");
    saveImageAction->setEnabled(false);
    connect(openImageAction, &QAction::triggered, this, &MainWindow::openImage);
    connect(saveImageAction, &QAction::triggered, this, &MainWindow::saveImage);
}

//setup Main image view
void MainWindow::setupImageView()
{
    imageView->setAlignment(Qt::AlignCenter);
    imageView->setVisible(false);
}

// Main Widget initialize
void MainWindow::setupMainWidget()
{
    // Main Layout
    QVBoxLayout *mainLayout = new QVBoxLayout;
    addSliderLayout(mainLayout, sharpenStrength, sharpenStrengthLabel);
    addSliderLayout(mainLayout, contrastAdaption, contrastAdaptionLabel);

    // Create Scroll Area
    scrollArea->setAlignment(Qt::AlignCenter);
    scrollArea->setWidgetResizable(true);
    scrollArea->setWidget(imageView);

    mainLayout->addWidget(scrollArea);
    // Central Widget
    QWidget *centralWidget = new QWidget;
    centralWidget->setLayout(mainLayout);
    setCentralWidget(centralWidget);
}

//add a Slider Horizontal layout (Slider and Label) into a Vertical layout
void MainWindow::addSliderLayout(QVBoxLayout *mainLayout, QSlider *slider, QLabel *label)
{
    QHBoxLayout *sliderLayout = new QHBoxLayout;
    sliderLayout->addWidget(label);
    sliderLayout->addWidget(slider);
    mainLayout->addLayout(sliderLayout);
}

//updates the Image label to show the passed-in QImage
void MainWindow::updateImageView(const QImage& image, const bool resetScale)
{
    QPixmap pixmap = QPixmap::fromImage(image);
    WidgetUtils::scalePixmap(pixmap, targetImageSize);
    resetScale ? imageView->setImage(pixmap) : imageView->updateImage(pixmap);
    scrollArea->setMinimumSize(pixmap.size() * 1.07);
}

//Open an image and display it to the user. Reinitialize CAS with the new dimensions
void MainWindow::openImage()
{
    const QString fileName = QFileDialog::getOpenFileName(this, "Open Image", "", imageDialogFilterText);
    if (fileName.isEmpty())
        return;
    QImageReader reader(fileName);
    reader.setAutoTransform(true);
    QImage readerImage = reader.read();
    if (readerImage.isNull())
	{
		QMessageBox::critical(this, "Open Image", "Failed to open the image.");
		return;
	}

	userImage = std::move(readerImage);
    sharpenedImage = QImage(userImage);

    //convert to RGBA interleaved format
    userImageHasAlpha = userImage.hasAlphaChannel();
    userImage = userImage.convertToFormat(QImage::Format_RGBA8888);
    //suppply image to re-initialize internal CAS memory
    CAS_supplyImage(casObj, userImage.constBits(), userImageHasAlpha, userImage.height(), userImage.width());

    // Only scale down if the image is larger than the target size
    updateImageView(userImage, true);
    WidgetUtils::setVisibility(true, imageView, sharpenStrength, contrastAdaption, sharpenStrengthLabel, contrastAdaptionLabel);
    saveImageAction->setEnabled(true);

    // Resize the window to fit the image and sliders
    adjustSize();

    //reset sliders
    sharpenStrength->setValue(0);
    contrastAdaption->setValue(100);
    
}

//Attempt to save the sharpened image
void MainWindow::saveImage()
{
    const QString fileName = QFileDialog::getSaveFileName(this, "Save Image", QString(), imageDialogFilterText);
    if (fileName.isEmpty())
        return;

    if (!sharpenedImage.save(fileName))
        QMessageBox::critical(this, "Save Image", "Failed to save the image.");
    else
        QMessageBox::information(this, "Save Image", "Image saved successfully.");
}

//event handler when a Slider is changed, triggers the CAS sharpening to occur and the display to show the new image
void MainWindow::sliderValueChanged()
{
    //don't calculate if parameters are (very close to) 0
    if (sharpenStrength->value() <= 0.001 || contrastAdaption->value() <= 0.001)
        return;

    //apply CAS CUDA from DLL and update UI
    const int sharpenedImageChannels = userImageHasAlpha ? 4 : 3;
    const auto sharpenedImageFormat = userImageHasAlpha ? QImage::Format_RGBA8888 : QImage::Format_RGB888;
    const uchar* casData = CAS_sharpenImage(casObj, 1, CLAMP(sharpenStrength->value()), CLAMP(contrastAdaption->value()));
    sharpenedImage = QImage(casData, userImage.width(), userImage.height(), userImage.width() * sharpenedImageChannels, sharpenedImageFormat);
    updateImageView(sharpenedImage, false);
}
