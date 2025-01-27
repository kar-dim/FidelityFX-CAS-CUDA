#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QSlider>
#include <QLabel>
#include <QImage>
#include <QVBoxLayout>
#include "..\FidelityFX-CAS-CUDA\include\CASLibWrapper.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void openImage();
    void saveImage();
    void sliderValueChanged();

private:
    const QString imageDialogFilterText { "Images (*.png *.jpg *.bmp *.webp *.tiff)" };
    void setupMenu();
    void setupSlider(QSlider *slider, QLabel *label, const int value);
    void setupImageView();
    void setupMainWidget();
    void addSliderLayout(QVBoxLayout *mainLayout, QSlider *slider, QLabel *label);
    void updateImageView(const QImage& image);

    QImage originalImage, sharpenedImage;
    QSlider *sharpenStrength, *contrastAdaption;
    QLabel *imageView, *sharpenStrengthLabel, *contrastAdaptionLabel;
    void* casObj;
    QAction *openImageAction, *saveImageAction;
    const QSize targetImageSize;
    bool originalImageAlpha;
};

#endif // MAINWINDOW_H
