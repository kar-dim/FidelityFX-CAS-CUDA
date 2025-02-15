#pragma once
#include <QLabel>
#include <QPixmap>
#include <QWheelEvent>
#include <QWidget>

class ZoomableLabel : public QLabel 
{
    QPixmap originalPixmap;
    double scaleFactor = 1.0;

public:
    explicit ZoomableLabel(QWidget* parent = nullptr);
    void setImage(const QPixmap& pixmap);
    void updateImage(const QPixmap& pixmap); //does not reset the zoom factor

protected:
    void wheelEvent(QWheelEvent* event) override;

private:
    void scaleImage();
};
