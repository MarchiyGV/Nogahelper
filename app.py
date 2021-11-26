from PyQt5.QtCore import (Qt, QSortFilterProxyModel, pyqtSignal, pyqtSlot, 
QThread, QModelIndex, QAbstractTableModel, QVariant, QPoint)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QInputDialog,
    QFileDialog,
    QAbstractItemView,
    QErrorMessage,
    QMessageBox,
    QShortcut,
    QDialog,
    QLabel
)
from PyQt5.QtGui import QCursor
import exception_hooks
import os
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib import cm, colors
import design
import design_export_png
import Model
import numpy as np
from matplotlib import pyplot as plt
class App(QMainWindow, design.Ui_MainWindow):
    ### Optimization
    Ca_min, Ca_max, Ca_num = 1, 500, 5
    Cb_min, Cb_max, Cb_num = 1, 500, 5
    Cab_min, Cab_max, Cab_num = 200, 700, 5
    S11_min = -20
    epsilon = 9
    Z = 5-15j
    Z0 = 50
    theta_min, theta_max, theta_step = 0.1, 30, 0.05
    opt_params_default = [Ca_min, Ca_max, Ca_num,
                          Cb_min, Cb_max, Cb_num,
                          Cab_min, Cab_max, Cab_num,
                          S11_min, Z,Z0, epsilon, 
                          theta_min, theta_max, theta_step]
    default_repr = 2 #0: Ca, 1: Cb, 2: Cab
    ### AFC
    Ca_step = 1
    Cb_step = 1
    Cab_step = 1
    Ca = (Ca_min+Ca_max)/2
    Cb = (Cb_min+Cb_max)/2
    Cab = (Cab_min+Cab_max)/2
    C_afc = [Ca, Cb, Cab]
    afc_params_default = [Ca_min,Cb_min,Cab_min,Ca_max,Cb_max,Cab_max,Ca_step,Cb_step,Cab_step]
    cmap_argmin = cm.viridis
    cmap_cont_width = cm.cool
    cmap_width = cm.viridis
    
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.model = Model.model(App.epsilon, App.Z0, App.Z)
        '''
        ###Optimization
        '''
        self.opt_inputs = [self.in_Ca_min,self.in_Ca_max,self.in_Ca_num,
                           self.in_Cb_min,self.in_Cb_max,self.in_Cb_num,
                           self.in_Cab_min,self.in_Cab_max,self.in_Cab_num,
                           self.in_S11_min,self.in_Z, self.in_Z0, self.in_epsilon,
                           self.in_theta_min,self.in_theta_max,self.in_theta_step]
        self.convert_from_Cz_slider = (lambda i: [App.Ca_min, 
                                                  App.Cb_min, 
                                                  App.Cab_min][App.default_repr])
        self.convert_to_Cz_slider = (lambda x: 0)
        fig = Figure()
        self.canvas_opt = FigureCanvas(fig)
        self.plot_opt_layout.addWidget(self.canvas_opt)
        self.toolbar = NavigationToolbar(self.canvas_opt, self.plot_opt_widget)
        self.plot_opt_layout.addWidget(self.toolbar)
        self.ax1_opt = self.canvas_opt.figure.add_subplot(121)
        self.ax2_opt = self.canvas_opt.figure.add_subplot(122)
        self.in_slider_Cz.valueChanged.connect(self.on_changed_Cz)
        self.in_Cz.valueChanged.connect(self.on_changed_Cz)
        for i, in_field in enumerate(self.opt_inputs):
            in_field.setText(str(App.opt_params_default[i]).replace('(', '').replace(')', ''))
        self.in_plot_repr.setCurrentIndex(App.default_repr)
        self.in_plot_repr.currentIndexChanged.connect(self.on_change_plot_opt_repr)
        self.in_plot_repr.currentIndexChanged.connect(self.init_plot_opt)
        self.btn_compute.clicked.connect(self.on_btn_compute)
        self.update_Cz()
        self.in_slider_Cz.blockSignals(True)
        self.in_Cz.blockSignals(True)
        self.check_contour.stateChanged.connect(self.plot_opt)
        self.check_autoscale.stateChanged.connect(self.on_changed_autoscale)
        self.last_save_path = os.getcwd()
        self.btn_save_opt.clicked.connect(self.save_opt)
        self.btn_open_opt.clicked.connect(self.open_opt)
        '''
        ###AFC
        '''
        self.afc_range = [[self.in_Ca_min_afc, self.in_Ca_max_afc],
                          [self.in_Cb_min_afc, self.in_Cb_max_afc],
                          [self.in_Cab_min_afc, self.in_Cab_max_afc]]
        self.afc_inputs = [self.in_Ca_min_afc,self.in_Cb_min_afc,self.in_Cab_min_afc,
                           self.in_Ca_max_afc,self.in_Cb_max_afc,self.in_Cab_max_afc]
        self.afc_sliders = [self.in_slider_Ca_afc,self.in_slider_Cb_afc,self.in_slider_Cab_afc]
        self.afc_dspinboxes = [self.in_Ca_afc,self.in_Cb_afc,self.in_Cab_afc]
        fig = Figure()
        self.canvas_afc = FigureCanvas(fig)
        self.plot_afc_layout.addWidget(self.canvas_afc)
        self.toolbar_afc = NavigationToolbar(self.canvas_afc, self.plot_afc_widget)
        self.plot_afc_layout.addWidget(self.toolbar_afc)
        self.ax_afc = self.canvas_afc.figure.add_subplot(111)
        self.ax_afc.set_aspect(0.5)
        self.convert_afc_from_slider = [lambda i: App.Ca_min, lambda i: App.Cb_min, lambda i: App.Cab_min]
        self.convert_afc_to_slider = [lambda x: 0, lambda x: 0, lambda x: 0]
        
        for i, in_field in enumerate(self.afc_inputs):
            in_field.setText(str(App.afc_params_default[i]))
            in_field.textChanged.connect(self.update_afc_sliders)
        for i, in_field in enumerate(self.afc_sliders):
            in_field.valueChanged.connect(self.on_changed_slider_afc)
        for i, in_field in enumerate(self.afc_dspinboxes):
            in_field.valueChanged.connect(self.on_changed_dspinbox_afc)
            in_field.blockSignals(True)
        self.update_afc_sliders()
        for i, in_field in enumerate(self.afc_dspinboxes):
            in_field.blockSignals(False)
        self.block_plotting_afc = True
        for i, in_field in enumerate(self.afc_dspinboxes):
            in_field.setValue(App.C_afc[i])
        self.block_plotting_afc = False
        self.init_plot_afc()
        self.actionExport_AFC_plot_as_png.triggered.connect(self.save_plot_afc)
        self.last_save_path_afc = ''
        self.check_autoscale_afc.stateChanged.connect(self.plot_afc)
    '''
    ###AFC
    '''
    @pyqtSlot(float)
    def on_changed_dspinbox_afc(self, val):
        sender = self.sender()
        i = self.afc_dspinboxes.index(sender)
        in_field = self.afc_sliders[i]
        in_field.setValue(self.convert_afc_to_slider[i](val))
        self.plot_afc()
        
    @pyqtSlot(int)
    def on_changed_slider_afc(self, n):
        sender = self.sender()
        i = self.afc_sliders.index(sender)
        in_field = self.afc_dspinboxes[i]
        in_field.setValue(self.convert_afc_from_slider[i](n))
        self.plot_afc()
        
    @pyqtSlot()
    def update_afc_sliders(self):
        sender = self.sender()
        for i in range(3):
            if sender in self.afc_range[i]:
                self.update_afc_slider(i)
        if sender is None:
            self.update_afc_slider(0)
            self.update_afc_slider(1)
            self.update_afc_slider(2)
            
    def update_afc_slider(self, i):
        _min = float(self.afc_range[i][0].text())
        _max = float(self.afc_range[i][1].text())
        num = 99
        step = (_max-_min)/num
        self.convert_afc_from_slider[i] = (lambda n: _min+(_max-_min)*(n/num))
        self.convert_afc_to_slider[i] = (lambda x: int(round(num*(x-_min)/(_max-_min))))
        self.afc_dspinboxes[i].setRange(_min, _max)
        self.afc_dspinboxes[i].setSingleStep(step)
        
    def init_plot_afc(self):
        if self.block_plotting_afc:
            return None
        self.canvas_afc.figure.clf()
        self.ax_afc = self.canvas_afc.figure.add_subplot(111)
        ax = self.ax_afc
        Ca = self.in_Ca_afc.value()
        Cb = self.in_Cb_afc.value()
        Cab = self.in_Cab_afc.value()
        S11_min = -20
        title, annotations, xs, theta, S11 = self.model.afc(Ca, Cb, Cab, S11_min)
        ax.set_title(title, fontsize=15)
        self.annotation_refs = []
        self.axvlines_refs = []
        for i in range(len(xs)):
            x = xs[i]
            annotation = annotations[i]
            self.axvlines_refs.append(ax.axvline(x=x, color='grey'))
            bbox = dict(boxstyle="round", fc="0.9")
            self.annotation_refs.append(ax.annotate(annotation, (x+6,S11_min), bbox=bbox))
        self.afc_plot_ref = ax.plot(theta, S11)
        ax.hlines(S11_min, theta[0], theta[-1], linestyle='--', color='black')
        ax.set_xlabel('$\Theta \degree$')
        ax.set_ylabel('$|S_{11}|, \,dB$')
        self.canvas_afc.figure.tight_layout()
        self.canvas_afc.draw()   
        
    def save_plot_afc(self):
        Ca = self.in_Ca_afc.value()
        Cb = self.in_Cb_afc.value()
        Cab = self.in_Cab_afc.value()
        exPopup = ExamplePopup(self, Ca, Cb, Cab)
        exPopup.setGeometry(100, 200, 1000, 500)
        exPopup.show()
        
    @pyqtSlot()
    def plot_afc(self):
        if self.block_plotting_afc:
            return None
        Ca = self.in_Ca_afc.value()
        Cb = self.in_Cb_afc.value()
        Cab = self.in_Cab_afc.value()
        S11_min = -20
        title, annotations, xs, theta, S11 = self.model.afc(Ca, Cb, Cab, S11_min)
        self.afc_plot_ref[0].set_ydata(S11)
        if self.check_autoscale_afc.checkState() == 2:
            self.ax_afc.relim()
            self.ax_afc.autoscale_view()
        for ref in self.annotation_refs:
            ref.remove()
        self.annotation_refs = []
        for ref in self.axvlines_refs:
            ref.remove()
        self.axvlines_refs = []
        for i in range(len(xs)):
            x = xs[i]
            annotation = annotations[i]
            self.axvlines_refs.append(self.ax_afc.axvline(x=x, color='grey'))
            bbox = dict(boxstyle="round", fc="0.9")
            _y, _ = self.ax_afc.get_ylim()
            self.annotation_refs.append(self.ax_afc.annotate(annotation, (x+6,_y+2), bbox=bbox))
        
        self.canvas_afc.draw()  
    '''
    ###Optimization
    '''
    @pyqtSlot()    
    def save_opt(self):
        fname = QFileDialog.getSaveFileName(self, 'Save file', self.last_save_path,'Text files (*.txt)')[0]
        if fname:
            self.last_save_path = os.path.dirname(fname)
            settings_opt = ''
            for input_field in self.opt_inputs:
                settings_opt+=(input_field.text()+'\n')
            with open(fname[0], 'w') as f:
                f.write(settings_opt)
                
    @pyqtSlot()    
    def open_opt(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', self.last_save_path, filter='Text files (*.txt)')[0]
        self.last_save_path = os.path.dirname(fname)
        if fname:
            with open(fname, 'r') as f:
                save = f.read().split('\n')
                save.remove('')
            for i, input_field in enumerate(self.opt_inputs):
                input_field.setText(save[i])
        else:
            pass             
        
    def Cz_range(self):
        i = self.in_plot_repr.currentIndex()
        Cz_min = float([self.in_Ca_min.text(), 
                        self.in_Cb_min.text(), 
                        self.in_Cab_min.text()][i])
        Cz_max = float([self.in_Ca_max.text(), 
                        self.in_Cb_max.text(), 
                        self.in_Cab_max.text()][i])
        Cz_num = int([self.in_Ca_num.text(), 
                      self.in_Cb_num.text(), 
                      self.in_Cab_num.text()][i])
        return Cz_min, Cz_max, Cz_num
        
    def update_Cz(self):
        val_old = self.convert_from_Cz_slider(self.in_slider_Cz.value())
        Cz_min, Cz_max, Cz_num = self.Cz_range()
        self.convert_from_Cz_slider = (lambda n: Cz_min+(Cz_max-Cz_min)*n/(Cz_num-1))
        self.convert_to_Cz_slider = (lambda Cz: int(round((Cz_num-1)*(Cz-Cz_min)/(Cz_max-Cz_min))))
        val_new = max(val_old, Cz_min)
        val_new = min(val_new, Cz_max)
        self.in_slider_Cz.blockSignals(True)
        self.in_slider_Cz.setRange(0, Cz_num-1)
        val_new_i = self.convert_to_Cz_slider(val_new)
        self.in_slider_Cz.setValue(val_new_i)
        self.in_slider_Cz.blockSignals(False)
        self.in_Cz.blockSignals(True)
        self.in_Cz.setRange(Cz_min, Cz_max)
        self.in_Cz.setSingleStep((Cz_max-Cz_min)/Cz_num)
        val_new = self.convert_from_Cz_slider(val_new_i)
        self.in_Cz.setValue(val_new)
        self.in_Cz.blockSignals(False)
        
    @pyqtSlot()
    def enable_opt(self):
        self.btn_compute.setEnabled(False)
        
    @pyqtSlot()
    def on_btn_compute(self):
        self.update_Cz()
        epsilon = float(self.in_epsilon.text())
        Z0 = float(self.in_Z0.text())
        Z = complex(self.in_Z.text())
        self.model = Model.model(epsilon, Z0, Z)
        self.model.started.connect(self.enable_opt)
        self.model.finished.connect(self.opt_finished)
        S11_min = float(self.in_S11_min.text())
        Ca_min = float(self.in_Ca_min.text())
        Ca_max = float(self.in_Ca_max.text())
        Ca_num = int(self.in_Ca_num.text())
        Cb_min = float(self.in_Cb_min.text())
        Cb_max = float(self.in_Cb_max.text())
        Cb_num = int(self.in_Cb_num.text())
        Cab_min = float(self.in_Cab_min.text())
        Cab_max = float(self.in_Cab_max.text())
        Cab_num = int(self.in_Cab_num.text())
        self.theta0_min = float(self.in_theta_min.text())
        self.theta0_max = float(self.in_theta_max.text())
        theta_step = float(self.in_theta_step.text())
        self.Ca0 = np.linspace(Ca_min, Ca_max, num=Ca_num)
        self.Cb0 = np.linspace(Cb_min, Cb_max, num=Cb_num)
        self.Cab0 = np.linspace(Cab_min, Cab_max, num=Cab_num)
        self.on_change_plot_opt_repr()
        theta0 = np.arange(self.theta0_min, self.theta0_max, theta_step)
        self.model.calc_map(S11_min, self.Ca0, self.Cb0, self.Cab0, theta0)

    @pyqtSlot()
    def opt_finished(self):
        self.btn_compute.setEnabled(True)
        self.argmin = self.model.argmin
        self.theta_min = self.model.theta_min
        self.min_w = self.model.min_w
        self.init_plot_opt()
        self.in_slider_Cz.blockSignals(False)
        self.in_Cz.blockSignals(False)
        
    @pyqtSlot()
    def on_changed_Cz(self):
        sender = self.sender()
        value=sender.value()
        if sender == self.in_slider_Cz:
            self.in_Cz.blockSignals(True)
            self.in_Cz.setValue(self.convert_from_Cz_slider(value))
            self.in_Cz.blockSignals(False)
            self.plot_opt(int(value))
        elif sender == self.in_Cz:
            self.in_slider_Cz.blockSignals(True)
            i = self.convert_to_Cz_slider(value)
            self.in_slider_Cz.setValue(i)
            self.in_slider_Cz.blockSignals(False)
            self.plot_opt(i)
            
    @pyqtSlot()
    def on_change_plot_opt_repr(self):
        k = self.in_plot_repr.currentIndex()
        if k==0:
            i, j = 1, 2
            self.x_label = 'C_b'
            self.y_label = 'C_{ab}'
            self.slice = lambda i: (i, slice(None), slice(None))
        if k==1:
            i, j = 0, 2
            self.x_label = 'C_a'
            self.y_label = 'C_{ab}'
            self.slice = lambda i: (slice(None), i, slice(None))
        if k==2:
            i, j = 0, 1
            self.x_label = 'C_a'
            self.y_label = 'C_b'
            self.slice = lambda i: (slice(None), slice(None), i)
        xyz = [self.Ca0, self.Cb0, self.Cab0]
        self.Cx = xyz[i]
        self.Cy = xyz[j]
        self.update_Cz()
    
    def init_plot_opt(self, i_Cz=None):
        if i_Cz is None:
            i_Cz = self.in_slider_Cz.value()
        self.canvas_opt.figure.clf()
        self.ax1_opt = self.canvas_opt.figure.add_subplot(121)
        self.ax2_opt = self.canvas_opt.figure.add_subplot(122)
        theta0_min = self.theta0_min
        theta0_max = self.theta0_max
        Cx = self.Cx
        Cy = self.Cy
        
        x_label = self.x_label
        y_label = self.y_label
        norm = colors.Normalize(vmin=theta0_min, vmax=theta0_max)
        ax = self.ax1_opt
        im = ax.contourf(Cx, Cy, np.transpose(self.theta_min[self.slice(i_Cz)]), cmap=self.cmap_argmin)
        self.ref_argmin_opt = im
        if self.check_contour.checkState()==2:
            self.ref_c_width_opt = ax.contour(Cx, Cy, np.transpose(self.min_w[self.slice(i_Cz)]), cmap=self.cmap_cont_width)
        ax.set_xlabel(f'${x_label}, pF$')
        ax.set_ylabel(f'${y_label}, pF$')
        ax.set_title('Argmin')
        im.set_clim(theta0_min, theta0_max)
        mpbl = cm.ScalarMappable(norm=norm, cmap=self.cmap_width)
        mpbl.set_array([])
        self.cbar_ax1_opt=ax.figure.colorbar(mpbl, ax=ax)
        self.cbar_ax1_opt.ax.set_title('$\Theta_{min} \degree$',fontsize=10)
    
        ax = self.ax2_opt
        norm = colors.Normalize(vmin=self.min_w.min(), vmax=self.min_w.max())
        im = ax.contourf(Cx, Cy, np.transpose(self.min_w[self.slice(i_Cz)]), cmap=self.cmap_width)
        self.ref_width_opt = im
        ax.set_xlabel(f'${x_label}, pF$')
        ax.set_ylabel(f'${y_label}, pF$')
        ax.set_title('Width of minima')
        im.set_clim(self.min_w.min(), self.min_w.max())
        mpbl = cm.ScalarMappable(norm=norm, cmap=self.cmap_width)
        mpbl.set_array([])
        self.cbar_ax2_opt=ax.figure.colorbar(mpbl, ax=ax)
        self.cbar_ax2_opt.ax.set_title('$\Delta\Theta_{min} \degree$',fontsize=10)
        self.canvas_opt.draw()
    
    @pyqtSlot()
    def plot_opt(self, i_Cz=None):
        if i_Cz is None:
            i_Cz = self.in_slider_Cz.value()
        theta0_min = self.theta0_min
        theta0_max = self.theta0_max
        Cx = self.Cx
        Cy = self.Cy
        for c in self.ref_argmin_opt.collections:
            c.remove() 
        try:
            for c in self.ref_c_width_opt.collections:
                c.remove() 
        except ValueError:#has been already deleted
            pass
        except AttributeError:#hasn't been created yet
            pass
        for c in self.ref_width_opt.collections:
            c.remove()    
        ax = self.ax1_opt
        self.ref_argmin_opt = ax.contourf(Cx, Cy, np.transpose(self.theta_min[self.slice(i_Cz)]), cmap=self.cmap_argmin)
        if self.check_contour.checkState()==2:
            self.ref_c_width_opt = ax.contour(Cx, Cy, np.transpose(self.min_w[self.slice(i_Cz)]), cmap=self.cmap_cont_width)
        ax = self.ax2_opt
        self.ref_width_opt = ax.contourf(Cx, Cy, np.transpose(self.min_w[self.slice(i_Cz)]), cmap=self.cmap_width)
        if self.check_autoscale.checkState() == 0:
            self.ref_argmin_opt.set_clim(theta0_min, theta0_max)
            self.ref_width_opt.set_clim(self.min_w.min(), self.min_w.max())
        else:
            self.cbar_ax1_opt.remove()
            self.cbar_ax2_opt.remove()
            self.cbar_ax1_opt=self.ax1_opt.figure.colorbar(self.ref_argmin_opt, ax=self.ax1_opt)
            self.cbar_ax1_opt.ax.set_title('$\Delta\Theta_{min} \degree$',fontsize=10)
            self.cbar_ax2_opt=self.ax2_opt.figure.colorbar(self.ref_width_opt, ax=self.ax2_opt)
            self.cbar_ax2_opt.ax.set_title('$\Delta\Theta_{min} \degree$',fontsize=10)
        self.canvas_opt.draw()
    
    @pyqtSlot(int)
    def on_changed_autoscale(self, state):
        if state == 0:
            self.init_plot_opt()
        else: 
            self.plot_opt()
            
class ExamplePopup(QDialog, design_export_png.Ui_Dialog):

    def __init__(self, parent, Ca, Cb, Cab):
        super().__init__(parent)
        self.setupUi(self)
        self.parent = parent
        fig = Figure()
        self.canvas = FigureCanvas(fig)
        self.plot_layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)
        self.Ca = Ca
        self.Cb = Cb
        self.Cab = Cab
        self.plot()
        self.aspect_ratio_slider.setRange(1, 10)
        self.aspect_ratio_slider.valueChanged.connect(self.plot)
        self.fname_edit.setText('AFC_plot')
        self.buttonBox.accepted.connect(self.accept)
        self.btn_open.clicked.connect(self.open)
        
    def open(self):
        fname = QFileDialog.getSaveFileName(self, 'Save file', self.parent.last_save_path_afc,'PNG (*.png)')[0]
        if fname:
            self.parent.last_save_path_afc = os.path.dirname(fname)
            self.fname_edit.setText(fname)
        
    def accept(self):
        fname = self.fname_edit.text()
        if fname:
            fname=fname.split('.')[0]
            self.canvas.print_png(fname+'.png')
        else:
            raise ValueError('file name should not be empty')
        self.close()
        
    @pyqtSlot(int)
    def plot(self, k=1):
        self.canvas.figure.clf()
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_aspect(k)
        
        ax = self.ax
        Ca = self.Ca
        Cb = self.Cb
        Cab = self.Cab
        S11_min = -20
        title, annotations, xs, theta, S11 = self.parent.model.afc(Ca, Cb, Cab, S11_min)
        ax.set_title(title, fontsize=15)
        ax.plot(theta, S11)
        ax.hlines(S11_min, theta[0], theta[-1], linestyle='--', color='black')
        for i in range(len(xs)):
            x = xs[i]
            annotation = annotations[i]
            ax.axvline(x=x, color='grey')
            bbox = dict(boxstyle="round", fc="0.9")
            _y, _ = ax.get_ylim()
            ax.annotate(annotation, (x+6,_y+2), bbox=bbox)
        
        ax.set_xlabel('$\Theta \degree$')
        ax.set_ylabel('$|S_{11}|, \,dB$')
        self.ax.figure.tight_layout()
        self.canvas.draw()   
        
        

if __name__ == '__main__': 
    import sys
    app = QApplication(sys.argv)  
    window = App() 
    window.show()  # Показываем окно
    sys.exit(app.exec_())